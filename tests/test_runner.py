"""
Test runner script for LitKG test suite.

Provides utilities for running tests with different configurations,
generating reports, and managing test environments.
"""

import pytest
import sys
import os
import argparse
from pathlib import Path
import subprocess
import json
import time
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestRunner:
    """Test runner with advanced configuration options."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run unit tests."""
        args = [
            "-m", "pytest",
            str(self.test_dir),
            "-v" if verbose else "-q",
            "-m", "unit or not slow",
            "--tb=short"
        ]
        
        if coverage:
            args.extend([
                "--cov=litkg",
                "--cov-report=html:test_reports/htmlcov",
                "--cov-report=term-missing",
                "--cov-report=json:test_reports/coverage.json"
            ])
        
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src")
        
        return subprocess.call([sys.executable] + args, cwd=self.project_root, env=env)
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        args = [
            "-m", "pytest",
            str(self.test_dir),
            "-v" if verbose else "-q",
            "-m", "integration",
            "--tb=long"
        ]
        
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src")
        
        return subprocess.call([sys.executable] + args, cwd=self.project_root, env=env)
    
    def run_slow_tests(self, verbose: bool = False) -> int:
        """Run slow tests."""
        args = [
            "-m", "pytest",
            str(self.test_dir),
            "-v" if verbose else "-q",
            "-m", "slow",
            "--tb=long"
        ]
        
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src")
        
        return subprocess.call([sys.executable] + args, cwd=self.project_root, env=env)
    
    def run_gpu_tests(self, verbose: bool = False) -> int:
        """Run GPU-specific tests."""
        args = [
            "-m", "pytest",
            str(self.test_dir),
            "-v" if verbose else "-q",
            "-m", "gpu",
            "--tb=short"
        ]
        
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src")
        
        return subprocess.call([sys.executable] + args, cwd=self.project_root, env=env)
    
    def run_specific_module(self, module: str, verbose: bool = False) -> int:
        """Run tests for a specific module."""
        test_file = self.test_dir / f"test_{module}.py"
        
        if not test_file.exists():
            print(f"Test file {test_file} does not exist")
            return 1
        
        args = [
            "-m", "pytest",
            str(test_file),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src")
        
        return subprocess.call([sys.executable] + args, cwd=self.project_root, env=env)
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = True) -> Dict[str, int]:
        """Run all test suites and return results."""
        results = {}
        
        print("🧪 Running LitKG Test Suite")
        print("=" * 50)
        
        # Unit tests
        print("\n📝 Running Unit Tests...")
        results["unit"] = self.run_unit_tests(verbose=verbose, coverage=coverage)
        
        # Integration tests
        print("\n🔗 Running Integration Tests...")
        results["integration"] = self.run_integration_tests(verbose=verbose)
        
        # GPU tests (if available)
        try:
            import torch
            if torch.cuda.is_available():
                print("\n🖥️  Running GPU Tests...")
                results["gpu"] = self.run_gpu_tests(verbose=verbose)
        except ImportError:
            print("\n⚠️  Skipping GPU tests (PyTorch not available)")
        
        return results
    
    def run_performance_tests(self) -> int:
        """Run performance benchmarks."""
        args = [
            "-m", "pytest",
            str(self.test_dir),
            "-v",
            "-m", "slow",
            "--benchmark-only",
            "--benchmark-json=test_reports/benchmark.json"
        ]
        
        return subprocess.call([sys.executable] + args, cwd=self.project_root)
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project": "LitKG-Integrate",
            "test_results": {},
            "coverage": {},
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(self.project_root)
            }
        }
        
        # Load coverage data if available
        coverage_file = self.reports_dir / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    report["coverage"] = json.load(f)
            except Exception as e:
                report["coverage"]["error"] = str(e)
        
        # Load benchmark data if available
        benchmark_file = self.reports_dir / "benchmark.json"
        if benchmark_file.exists():
            try:
                with open(benchmark_file) as f:
                    report["benchmarks"] = json.load(f)
            except Exception as e:
                report["benchmarks"] = {"error": str(e)}
        
        # Save report
        report_file = self.reports_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def check_test_environment(self) -> Dict[str, bool]:
        """Check test environment setup."""
        checks = {}
        
        # Check Python version
        checks["python_version"] = sys.version_info >= (3, 9)
        
        # Check required packages
        required_packages = [
            "pytest", "torch", "transformers", "networkx", 
            "pandas", "numpy", "scikit-learn"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                checks[f"package_{package}"] = True
            except ImportError:
                checks[f"package_{package}"] = False
        
        # Check GPU availability
        try:
            import torch
            checks["gpu_available"] = torch.cuda.is_available()
        except ImportError:
            checks["gpu_available"] = False
        
        # Check test data directory
        checks["test_data_dir"] = (self.test_dir / "test_data").exists()
        
        return checks
    
    def setup_test_environment(self):
        """Set up test environment."""
        print("🔧 Setting up test environment...")
        
        # Create test data directory
        test_data_dir = self.test_dir / "test_data"
        test_data_dir.mkdir(exist_ok=True)
        
        # Create test models directory
        test_models_dir = self.test_dir / "test_models"
        test_models_dir.mkdir(exist_ok=True)
        
        # Set environment variables for testing
        os.environ["LITKG_TEST_MODE"] = "true"
        os.environ["LITKG_DATA_DIR"] = str(test_data_dir)
        os.environ["LITKG_MODELS_DIR"] = str(test_models_dir)
        
        print("✅ Test environment setup complete")
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        print("🧹 Cleaning up test environment...")
        
        # Remove test environment variables
        test_env_vars = ["LITKG_TEST_MODE", "LITKG_DATA_DIR", "LITKG_MODELS_DIR"]
        for var in test_env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Clean up temporary test files
        temp_files = [
            self.project_root / ".coverage",
            self.project_root / ".pytest_cache"
        ]
        
        for temp_file in temp_files:
            if temp_file.exists():
                if temp_file.is_file():
                    temp_file.unlink()
                else:
                    import shutil
                    shutil.rmtree(temp_file)
        
        print("✅ Test environment cleanup complete")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="LitKG Test Runner")
    parser.add_argument(
        "--suite",
        choices=["unit", "integration", "slow", "gpu", "all"],
        default="unit",
        help="Test suite to run"
    )
    parser.add_argument(
        "--module",
        help="Specific module to test (e.g., 'utils', 'phase1')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--setup-env",
        action="store_true",
        help="Set up test environment"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up test environment"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check test environment"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate test report"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Handle environment commands
    if args.setup_env:
        runner.setup_test_environment()
        return 0
    
    if args.cleanup:
        runner.cleanup_test_environment()
        return 0
    
    if args.check_env:
        checks = runner.check_test_environment()
        print("\n🔍 Test Environment Check:")
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check}: {status}")
        return 0 if all(checks.values()) else 1
    
    if args.report:
        report = runner.generate_test_report()
        print(f"\n📊 Test report generated: {runner.reports_dir / 'test_report.json'}")
        return 0
    
    # Set up environment for test run
    runner.setup_test_environment()
    
    try:
        # Run tests
        if args.module:
            exit_code = runner.run_specific_module(args.module, verbose=args.verbose)
        elif args.suite == "unit":
            exit_code = runner.run_unit_tests(
                verbose=args.verbose,
                coverage=not args.no_coverage
            )
        elif args.suite == "integration":
            exit_code = runner.run_integration_tests(verbose=args.verbose)
        elif args.suite == "slow":
            exit_code = runner.run_slow_tests(verbose=args.verbose)
        elif args.suite == "gpu":
            exit_code = runner.run_gpu_tests(verbose=args.verbose)
        elif args.suite == "all":
            results = runner.run_all_tests(
                verbose=args.verbose,
                coverage=not args.no_coverage
            )
            exit_code = max(results.values()) if results else 0
        
        # Generate report
        if not args.no_coverage:
            runner.generate_test_report()
        
        return exit_code
    
    finally:
        # Clean up environment
        runner.cleanup_test_environment()


if __name__ == "__main__":
    sys.exit(main())