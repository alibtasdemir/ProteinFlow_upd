import os
import hydra
from omegaconf import DictConfig
import torch
import logging
import shutil

from models.proteinflow_clf_wrapperv2 import ProteinFlowModulev2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GPU_ID = 3  # Change this to the GPU you want to use

def setup_test_data():
    """Ensure test data directory exists and contains example PDB."""
    # Use 6F57_A.pdb from current directory
    example_pdb = "6F57_A.pdb"
    if not os.path.exists(example_pdb):
        logger.error(f"PDB file not found at {example_pdb}")
        raise FileNotFoundError(f"Missing {example_pdb}")
    
    return example_pdb

def setup_device():
    """Setup GPU device."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return torch.device("cpu")
    
    if GPU_ID >= torch.cuda.device_count():
        logger.warning(f"GPU {GPU_ID} not available, using GPU 0")
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{GPU_ID}")
    
    torch.cuda.set_device(device)
    logger.info(f"Using GPU {GPU_ID}: {torch.cuda.get_device_name(device)}")
    return device

def setup_model(cfg: DictConfig, device: torch.device) -> ProteinFlowModulev2:
    """Initialize and load the model.
    
    Args:
        cfg: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded and configured model
    """
    logger.info("Initializing model...")
    model = ProteinFlowModulev2(cfg)
    
    # Load checkpoint
    checkpoint_path = "ckpt/se3-fm/dnmt-full-unconditioned/2024-09-26_00-42-38/epoch29.ckpt"
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint to specified device
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model = model.to(device)
    model.eval()
    
    # Set device for interpolant
    model.interpolant.set_device(device)
    
    # Set inference config
    model._infer_cfg = cfg.inference
    
    logger.info(f"Model and interpolant loaded on {device}")
    return model

def test_central_residues(model: ProteinFlowModulev2, example_pdb: str, output_dir: str):
    """Test fixing central residues.
    
    Args:
        model: Loaded model
        example_pdb: Path to example PDB file
        output_dir: Directory to save outputs
    """
    logger.info("Test 1: Fixing central residues...")
    try:
        # Fix residues 2-4 in the 5-residue helix
        central_residues = [2, 3, 4]
        samples = model.sample_with_fixed_residues(
            pdb_path=example_pdb,
            fixed_residues=central_residues,
            num_samples=2,
            temperature=0.8,
            output_dir=output_dir
        )
        logger.info(f"Generated samples with fixed central residues: {samples}")
        
        # Validate the samples
        for i, sample_path in enumerate(samples):
            quality_metrics = model.evaluate_structure_quality(
                sample_path, 
                reference_pdb_path=example_pdb,
                fixed_residues=central_residues
            )
            logger.info(f"Sample {i} quality metrics: {quality_metrics}")
            
    except Exception as e:
        logger.error(f"Error in central residue test: {e}")
        logger.exception("Detailed traceback:")

def test_terminal_residues(model: ProteinFlowModulev2, example_pdb: str, output_dir: str):
    """Test fixing terminal residues.
    
    Args:
        model: Loaded model
        example_pdb: Path to example PDB file
        output_dir: Directory to save outputs
    """
    logger.info("Test 2: Fixing terminal residues...")
    try:
        # Fix first and last residues
        terminal_residues = [1, 5]
        samples = model.sample_with_fixed_residues(
            pdb_path=example_pdb,
            fixed_residues=terminal_residues,
            num_samples=2,
            temperature=0.8,
            output_dir=output_dir
        )
        logger.info(f"Generated samples with fixed terminals: {samples}")
        
        for i, sample_path in enumerate(samples):
            quality_metrics = model.evaluate_structure_quality(
                sample_path, 
                reference_pdb_path=example_pdb,
                fixed_residues=terminal_residues
            )
            logger.info(f"Sample {i} quality metrics: {quality_metrics}")
            
    except Exception as e:
        logger.error(f"Error in terminal residue test: {e}")
        logger.exception("Detailed traceback:")

def test_temperature_sweep(model: ProteinFlowModulev2, example_pdb: str, output_dir: str):
    """Test different temperature values.
    
    Args:
        model: Loaded model
        example_pdb: Path to example PDB file
        output_dir: Directory to save outputs
    """
    logger.info("Test 3: Temperature sensitivity analysis...")
    temperatures = [0.5, 1.0, 1.5]
    fixed_residues = [2, 3]  # Fix middle residues
    
    for temp in temperatures:
        try:
            samples = model.sample_with_fixed_residues(
                pdb_path=example_pdb,
                fixed_residues=fixed_residues,
                num_samples=1,
                temperature=temp,
                output_dir=output_dir
            )
            logger.info(f"Sample at temperature {temp} saved at: {samples}")
            
            # Save and analyze sample
            quality_metrics = model.evaluate_structure_quality(
                samples[0], 
                reference_pdb_path=example_pdb,
                fixed_residues=fixed_residues
            )
            logger.info(f"Quality metrics at temperature {temp}: {quality_metrics}")
            
        except Exception as e:
            logger.error(f"Error at temperature {temp}: {e}")
            logger.exception("Detailed traceback:")

def test_classifier_guidance(model: ProteinFlowModulev2, example_pdb: str, output_dir: str, cfg: DictConfig):
    """Test classifier-guided sampling.
    
    Args:
        model: Loaded model
        example_pdb: Path to example PDB file
        output_dir: Directory to save outputs
        cfg: Configuration dictionary
    """
    logger.info("Test 4: Classifier-guided sampling...")
    try:
        # Debug logging for config
        logger.info("Checking inference config...")
        logger.info(f"Full inference config: {cfg.inference}")
        
        if not hasattr(cfg, 'inference'):
            logger.error("No inference config found in cfg")
            return
            
        if not hasattr(cfg.inference, 'classifier'):
            logger.error("No classifier config found in cfg.inference")
            return
            
        logger.info(f"Classifier config: {cfg.inference.classifier}")
        
        # Load classifier if available
        if hasattr(cfg.inference.classifier, 'ckpt_path'):
            logger.info(f"Loading classifier from: {cfg.inference.classifier.ckpt_path}")
            if not os.path.exists(cfg.inference.classifier.ckpt_path):
                logger.error(f"Classifier checkpoint not found at: {cfg.inference.classifier.ckpt_path}")
                return
                
            try:
                model.load_classifiers(cfg.inference.classifier)
                logger.info("Classifier loaded successfully")
            except Exception as e:
                logger.error(f"Error loading classifier: {e}")
                logger.exception("Detailed traceback:")
                return
            
            # Test different guidance scales
            guidance_scales = [0.2]
            # Use a smaller set of residues that are within the valid range for 6F57_A.pdb
            fixed_residues = [639, 640, 641, 642, 643, 644, 645, 646, 647, 664, 665, 686, 687, 688, 888, 889, 890, 891, 892, 893, 894, 895]
            
            for scale in guidance_scales:
                logger.info(f"Testing with guidance scale {scale}...")
                try:
                    samples = model.sample_with_fixed_residues(
                        pdb_path=example_pdb,
                        fixed_residues=fixed_residues,
                        num_samples=1,
                        temperature=0.8,
                        output_dir=os.path.join(output_dir, f"scale_{scale}"),
                        clf_model=model.cls_model,
                        guidance_scale=scale,
                        target_class=1
                    )
                    
                    # Evaluate samples
                    for i, sample_path in enumerate(samples):
                        quality_metrics = model.evaluate_structure_quality(
                            sample_path,
                            reference_pdb_path=example_pdb,
                            fixed_residues=fixed_residues
                        )
                        logger.info(f"Sample {i} with guidance scale {scale} quality metrics: {quality_metrics}")
                except Exception as e:
                    logger.error(f"Error during sampling with scale {scale}: {e}")
                    logger.exception("Detailed traceback:")
        else:
            logger.warning("No classifier checkpoint path found in config")
            
    except Exception as e:
        logger.error(f"Error in classifier-guided sampling test: {e}")
        logger.exception("Detailed traceback:")

@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def test_conditional_sampling(cfg: DictConfig):
    """Test the conditional sampling functionality.
    
    This script:
    1. Loads a pretrained model
    2. Uses a small example alpha helix
    3. Tests different conditional sampling scenarios
    4. Saves and validates the results
    """
    # Setup device
    device = setup_device()
    
    # Setup test environment
    logger.info("Setting up test environment...")
    example_pdb = setup_test_data()
    
    # Create output directories
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different test cases
    central_dir = os.path.join(output_dir, "central_fixed")
    terminal_dir = os.path.join(output_dir, "terminal_fixed")
    temperature_dir = os.path.join(output_dir, "temperature_sweep")
    classifier_dir = os.path.join(output_dir, "classifier_guided")
    os.makedirs(central_dir, exist_ok=True)
    os.makedirs(terminal_dir, exist_ok=True)
    os.makedirs(temperature_dir, exist_ok=True)
    os.makedirs(classifier_dir, exist_ok=True)
    
    # Initialize model
    model = setup_model(cfg, device)
    
    # Run test cases
    # test_central_residues(model, example_pdb, central_dir)
    # test_terminal_residues(model, example_pdb, terminal_dir)
    # test_temperature_sweep(model, example_pdb, temperature_dir)
    test_classifier_guidance(model, example_pdb, classifier_dir, cfg)
    
    logger.info(f"All test outputs saved to {output_dir}")
    logger.info("Testing complete!")

if __name__ == "__main__":
    test_conditional_sampling() 