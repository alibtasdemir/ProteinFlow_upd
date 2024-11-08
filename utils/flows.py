import torch
from utils import so3Utils as su
from scipy.spatial.transform import Rotation
from utils import all_atom
import copy
from scipy.optimize import linear_sum_assignment
from utils.modelUtils import batch_align_structures, to_numpy

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch * num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
            rotmats_t * diffuse_mask[..., None, None]
            + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = su.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]

    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch * num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = su.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
                rotmats_t * res_mask[..., None, None]
                + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return su.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, self._device) * NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        batch = {
            'res_mask': res_mask,
        }

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj
    
    
    def guidance_score(self, structure, classifier, target_class=1):
        # Inputs:
        # structure -> Dictionary, structure dict includes trans_t, rotmats_t, t, etc.
        # classifier -> Model, Classifier model
        # target_class -> int, integer value for target class
        for key in structure.keys():
            structure[key] = structure[key].requires_grad_(True)
        
        # structure['trans_t'] = structure['trans_t'].requires_grad_(True)
        # structure['rotmats_t'] = structure['rotmats_t'].requires_grad_(True)
        print(f"structure[trans_t] requires grad?: {structure["trans_t"].grad_fn}")
        print(f"structure[rotmats_t] requires grad?: {structure["rotmats_t"].grad_fn}")
        # Get predictions
        classifier.train()
        torch.set_grad_enabled(True)
        prediction = classifier(structure)
        print("The output of the classifier:")
        print(prediction)
        # Calculate class probabilities
        prediction = torch.nn.functional.softmax(prediction, dim=1)
        print("Inside guidance_score, prediction;")
        print(prediction)
        print(prediction[0])
        print(f"Model output requires grad: {prediction.grad_fn}")
        # Get only target signal
        score = prediction[0][target_class]
        print(f"Score requires grad: {score.grad_fn}")
        return score
    
    
    def compute_guidance_gradient(self, structure, classifier, target_class):
        # Ensure gradients are enabled for the structure
        structure['trans_t'] = structure['trans_t'].requires_grad_(True)
        structure['rotmats_t'] = structure['rotmats_t'].requires_grad_(True)
        
        # Get the score for the target class
        score = self.guidance_score(structure, classifier, target_class)
        # Compute gradients
        score.backward()
        translation_gradient = structure['trans_t'].grad.detach()
        rotation_gradient = structure['rotmats_t'].grad.detach()
        
        # Clear gradients
        structure['trans_t'].grad = None
        structure['rotmats_t'].grad = None
        return translation_gradient, rotation_gradient
    
    
    def sample_clf(
            self,
            num_batch,
            num_res,
            model,
            clf_model,
            guidance_scale=0.2,
            target_class=1,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, self._device) * NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        batch = {
            'res_mask': res_mask,
        }

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            
            # Create a fake batch
            next_batch = copy.deepcopy(batch)
            next_batch['trans_t'] = pred_trans_1
            next_batch['rotmats_t'] = pred_rotmats_1
            with torch.enable_grad():
                next_batch['trans_t'] = next_batch['trans_t'].requires_grad_(True)
                next_batch['rotmats_t'] = next_batch['rotmats_t'].requires_grad_(True)
                
                grad = self.compute_guidance_gradient(next_batch, clf_model.model, target_class=target_class)
                print(grad)
                
                """
                cls_logits = clf_model.model(xt_)
                print(cls_logits)
                print(cls_logits.shape)
                cls_logits = cls_logits.requires_grad_(True)
                # print("Classifier outputs")
                # print(cls_logits)
                target_tensor = torch.tensor([target_class], device=self._device)
                # probabilities = torch.nn.functional.softmax(cls_logits, dim=1)
                classifier_loss = torch.nn.functional.cross_entropy(cls_logits, target_tensor)
                classifier_loss = classifier_loss.requires_grad_(True)
                # print(classifier_loss)
                classifier_loss.backward()
                print(classifier_loss)
                print()
                print(next_batch['trans_t'])
                cls_score_wrt_trans = - torch.autograd.grad(classifier_loss, [next_batch['trans_t']])[0]
                # cls_score = - torch.autograd.grad(loss, [next_batch])[0]
                """
            
            pred_trans_1 = pred_trans_1 + guidance_scale * grad
            pred_rotmats_1 = pred_rotmats_1 + guidance_scale * grad
            
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2
            

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj
