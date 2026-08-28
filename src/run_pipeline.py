import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

class PipelineRunner:

    def __init__(
        self,
        config_name: str = "chr22_5kb_h2",
        resolution: int = 5000,
        bin_size: int = 5000,
        num_threads: int = 1,
        domain_list: Optional[Path] = None,
        hic_matrix: Optional[Path] = None,
        mapping_file: Optional[Path] = None,
        run_name: Optional[str] = None,
    ):
        self.config_name = config_name
        self.resolution = resolution
        self.bin_size = bin_size
        self.num_threads = num_threads

        self.domain_list_override = domain_list
        self.hic_matrix_override = hic_matrix
        self.mapping_file_override = mapping_file

        if run_name is None:
            config_path = Path(config_name)
            self.run_name = config_path.name if config_path.parent != Path(".") else config_name
        else:
            self.run_name = run_name

        self.base_dir = Path(__file__).parent.absolute()

        self.setup_paths()

        self.create_directories()

    def setup_paths(self):
        self.preprocessing_dir = self.base_dir / "preprocessing" / self.config_name

        parts = self.run_name.split('_')
        chrom = parts[0]
        resolution_str = parts[1] if len(parts) > 1 else str(self.resolution // 1000) + "kb"

        self.domain_list = self.domain_list_override or (self.preprocessing_dir / f"{chrom}_domainlist_arrowhead.txt")
        self.hic_matrix = self.hic_matrix_override or (self.preprocessing_dir / f"{chrom}_{resolution_str}.txt")
        self.mapping_file = self.mapping_file_override or (
            self.preprocessing_dir / f"{chrom}_{resolution_str}_coordinate_mapping.txt"
        )

        self.phase1_output_dir = self.preprocessing_dir / "output" / self.run_name
        self.phase1_raw_output = self.phase1_output_dir / f"{self.run_name}_domain_interactions_raw.txt"
        self.phase1_3col_output = self.phase1_output_dir / f"{self.run_name}_domain_interactions.txt"
        self.phase1_filtered_domains = self.phase1_output_dir / f"{self.run_name}_filtered_domains.txt"

        self.phase2_script = self.base_dir / "main.py"
        self.phase2_embedding_output = self.base_dir / "Outputs" / f"{self.run_name}_domain_interactions_embeddings.txt"
        self.phase2_weights_output = self.base_dir / "Outputs" / f"{self.run_name}_domain_interactions_weights.pt"

        self.phase3_script = self.base_dir / "intra_domains" / "domain_structures.py"
        config_path = Path(self.config_name)
        experiment_subdir = config_path.parent if config_path.parent != Path(".") else None
        if experiment_subdir is None:
            self.phase3_output_dir = self.base_dir / "intra_domains" / "outputs" / self.run_name
        else:
            self.phase3_output_dir = self.base_dir / "intra_domains" / "outputs" / experiment_subdir / self.run_name

        self.phase1_script = self.base_dir / "preprocessing" / "domain_interactions.py"

    def create_directories(self):
        directories = [
            self.phase1_output_dir,
            self.base_dir / "Outputs",
            self.phase3_output_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Ensured directory exists: {directory}")

    def check_inputs(self) -> bool:
        required_inputs = [
            self.domain_list,
            self.hic_matrix,
            self.mapping_file,
            self.phase1_script,
            self.phase2_script,
            self.phase3_script,
        ]

        missing = [f for f in required_inputs if not f.exists()]

        if missing:
            print("❌ Missing required input files:")
            for f in missing:
                print(f"   - {f}")
            return False

        print("✓ All required input files found")
        print("\nResolved inputs:")
        print(f"  - Domain list : {self.domain_list}")
        print(f"  - Hi-C matrix  : {self.hic_matrix}")
        print(f"  - Mapping file : {self.mapping_file}")
        print("Resolved outputs:")
        print(f"  - Phase 1 out dir: {self.phase1_output_dir}")
        print(f"  - Phase 3 out dir: {self.phase3_output_dir}")
        return True

    def run_command(self, cmd: list, phase_name: str) -> bool:
        print(f"\n{'='*70}")
        print(f"Running {phase_name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                cwd=self.base_dir,
                capture_output=False,
            )
            print(f"\n✓ {phase_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {phase_name} failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print(f"\n❌ Command not found. Make sure Python is in your PATH.")
            return False

    def run_phase1(self) -> bool:
        cmd = [
            sys.executable,
            str(self.phase1_script),
            "--domain-list", str(self.domain_list),
            "--hic-matrix", str(self.hic_matrix),
            "--mapping", str(self.mapping_file),
            "--out-raw", str(self.phase1_raw_output),
            "--resolution", str(self.resolution),
            "--out-3col", str(self.phase1_3col_output),
            "--out-filtered-domain-list-combined", str(self.phase1_filtered_domains),
        ]

        success = self.run_command(cmd, "Phase 1: Domain Interactions Extraction")

        if success:
            if not self.phase1_3col_output.exists():
                print(f"❌ Expected output file not found: {self.phase1_3col_output}")
                return False
            if not self.phase1_filtered_domains.exists():
                print(f"❌ Expected output file not found: {self.phase1_filtered_domains}")
                return False

        return success

    def run_phase2(self) -> bool:
        if not self.phase1_3col_output.exists():
            print(f"❌ Phase 1 output not found: {self.phase1_3col_output}")
            print("   Please run Phase 1 first.")
            return False

        cmd = [
    	sys.executable,
    	"-m", "main",
    	str(self.phase1_3col_output),
	]

        success = self.run_command(cmd, "Phase 2")

        if success:
            if not self.phase2_embedding_output.exists():
                print(f"⚠ Warning: Expected embedding file not found: {self.phase2_embedding_output}")
                print("   The script may have created it with a different name. Continuing...")
            if not self.phase2_weights_output.exists():
                print(f"⚠ Warning: Expected weights file not found: {self.phase2_weights_output}")
                print("   The script may have created it with a different name. Continuing...")

        return success

    def run_phase3(self) -> bool:
        if not self.phase1_filtered_domains.exists():
            print(f"❌ Phase 1 output not found: {self.phase1_filtered_domains}")
            return False

        embedding_file = self.phase2_embedding_output
        if not embedding_file.exists():
            data_dir = self.base_dir / "Outputs"
            if data_dir.exists():
                matching_files = list(data_dir.glob(f"*{self.config_name}*embeddings*.txt"))
                if matching_files:
                    embedding_file = matching_files[0]
                    print(f"ℹ Using embedding file: {embedding_file}")
                else:
                    print(f"⚠ Warning: Embedding file not found. Using expected path: {embedding_file}")

        weights_file = self.phase2_weights_output
        if not weights_file.exists():
            outputs_dir = self.base_dir / "Outputs"
            if outputs_dir.exists():
                matching_files = list(outputs_dir.glob(f"*{self.config_name}*weights*.pt"))
                if matching_files:
                    weights_file = matching_files[0]
                    print(f"ℹ Using weights file: {weights_file}")
                else:
                    print(f"⚠ Warning: Weights file not found. Using expected path: {weights_file}")

        cmd = [
            sys.executable,
            str(self.phase3_script),
            "--hic_file", str(self.hic_matrix),
            "--domain_file", str(self.phase1_filtered_domains),
            "--mapping_file", str(self.mapping_file),
            "--embedding_file", str(embedding_file),
            "--output_dir", str(self.phase3_output_dir),
            "--pretrained_model_path", str(weights_file),
            "--num_threads", str(self.num_threads),
            "--bin_size", str(self.bin_size),
        ]

        return self.run_command(cmd, "Phase 3: Domain Structures Generation")

    def run_all(self) -> bool:
        print(f"\n{'#'*70}")
        print(f"# Configuration: {self.config_name}")
        print(f"# Resolution: {self.resolution} bp")
        print(f"# Bin Size: {self.bin_size} bp")
        print(f"# Threads: {self.num_threads}")
        print(f"{'#'*70}\n")

        if not self.check_inputs():
            return False

        phases = [
            ("Phase 1", self.run_phase1),
            ("Phase 2", self.run_phase2),
            ("Phase 3", self.run_phase3),
        ]

        for phase_name, phase_func in phases:
            if not phase_func():
                print(f"\n❌ Pipeline stopped at {phase_name}")
                return False

        print(f"\n{'#'*70}")
        print("# Pipeline completed successfully! ✓")
        print(f"{'#'*70}\n")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Run the HiCGAT pipeline automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (chr17_5kb_h3, 5000bp resolution)
  python run_pipeline.py
  
  # Run with custom configuration
  python run_pipeline.py --config chr17_10kb_h3 --resolution 10000
  
  # Run with custom threads
  python run_pipeline.py --threads 50
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="chr17_5kb_h3",
        help=(
            "Configuration name used for legacy input filename inference under preprocessing/<config>/ "
            "and default output naming. Default: chr17_5kb_h3"
        ),
    )

    parser.add_argument(
        "--domain-list",
        type=str,
        default=None,
        help=(
            "(Optional) Explicit path to the domain list file. "
            "If provided, does not require legacy naming under preprocessing/<config>/."
        ),
    )
    parser.add_argument(
        "--hic-matrix",
        type=str,
        default=None,
        help=(
            "(Optional) Explicit path to the Hi-C matrix file (txt). "
            "If provided, does not require legacy naming under preprocessing/<config>/."
        ),
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default=None,
        help=(
            "(Optional) Explicit path to the coordinate mapping file. "
            "If provided, does not require legacy naming under preprocessing/<config>/."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="(Optional) Name used to label output folders/files (default: --config).",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=5000,
        help="Resolution in base pairs. Default: 5000"
    )

    parser.add_argument(
        "--bin-size",
        type=int,
        default=5000,
        help="Bin size for domain structures. Default: 5000"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=25,
        help="Number of threads for Phase 3. Default: 25"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate resolved inputs and exit (does not run Phase 1/2/3).",
    )

    args = parser.parse_args()

    domain_list = Path(args.domain_list).expanduser().resolve() if args.domain_list else None
    hic_matrix = Path(args.hic_matrix).expanduser().resolve() if args.hic_matrix else None
    mapping_file = Path(args.mapping_file).expanduser().resolve() if args.mapping_file else None

    runner = PipelineRunner(
        config_name=args.config,
        resolution=args.resolution,
        bin_size=args.bin_size,
        num_threads=args.threads,
        domain_list=domain_list,
        hic_matrix=hic_matrix,
        mapping_file=mapping_file,
        run_name=args.run_name,
    )

    if args.check_only:
        ok = runner.check_inputs()
        sys.exit(0 if ok else 1)

    success = runner.run_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
