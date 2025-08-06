#!/usr/bin/env python3
"""
Command-line interface for LitKG-Integrate.

This module provides a unified CLI for all LitKG operations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from .utils.logging import setup_logging
from .utils.config import load_config


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="litkg",
        description="LitKG-Integrate: Literature-Augmented Knowledge Graph Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  litkg setup                           # Setup models and environment
  litkg phase1 --query "BRCA1 cancer"  # Run Phase 1 pipeline
  litkg literature --query "TP53"      # Process literature only
  litkg kg --sources civic,tcga        # Process knowledge graphs only
  litkg link --input literature.json   # Perform entity linking only
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: logs/litkg.log)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup models and environment")
    setup_parser.add_argument(
        "--force", 
        action="store_true",
        help="Force reinstall of models"
    )
    
    # Phase 1 command
    phase1_parser = subparsers.add_parser("phase1", help="Run complete Phase 1 pipeline")
    phase1_parser.add_argument(
        "--queries",
        nargs="+",
        default=["BRCA1 breast cancer", "TP53 lung cancer"],
        help="Literature search queries"
    )
    phase1_parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum results per query"
    )
    phase1_parser.add_argument(
        "--output",
        type=str,
        default="data/processed/phase1_results.json",
        help="Output file path"
    )
    
    # Literature processing command
    lit_parser = subparsers.add_parser("literature", help="Process literature only")
    lit_parser.add_argument(
        "--query",
        required=True,
        help="PubMed search query"
    )
    lit_parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum number of articles to process"
    )
    lit_parser.add_argument(
        "--output",
        type=str,
        default="data/processed/literature_results.json",
        help="Output file path"
    )
    
    # Knowledge graph processing command
    kg_parser = subparsers.add_parser("kg", help="Process knowledge graphs only")
    kg_parser.add_argument(
        "--sources",
        nargs="+",
        choices=["civic", "tcga", "cptac"],
        default=["civic", "tcga", "cptac"],
        help="Knowledge graph sources to process"
    )
    kg_parser.add_argument(
        "--output",
        type=str,
        default="data/processed/knowledge_graph.json",
        help="Output file path"
    )
    
    # Entity linking command
    link_parser = subparsers.add_parser("link", help="Perform entity linking only")
    link_parser.add_argument(
        "--literature",
        required=True,
        help="Literature results JSON file"
    )
    link_parser.add_argument(
        "--kg",
        required=True,
        help="Knowledge graph JSON file"
    )
    link_parser.add_argument(
        "--output",
        type=str,
        default="data/processed/entity_links.json",
        help="Output file path"
    )
    
    return parser


def cmd_setup(args) -> int:
    """Handle setup command."""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Get the project root (assuming we're in src/litkg/)
        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "setup_models.py"
        
        if not script_path.exists():
            print(f"Setup script not found at {script_path}", file=sys.stderr)
            return 1
        
        # Run the setup script
        result = subprocess.run([
            sys.executable, str(script_path)
        ], cwd=str(project_root))
        
        return result.returncode
        
    except Exception as e:
        print(f"Error running setup: {e}", file=sys.stderr)
        return 1


def cmd_phase1(args) -> int:
    """Handle phase1 command."""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Get the project root
        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "phase1_integration.py"
        
        if not script_path.exists():
            print(f"Phase1 script not found at {script_path}", file=sys.stderr)
            return 1
        
        # Build command arguments
        cmd_args = [sys.executable, str(script_path)]
        
        if args.queries:
            cmd_args.extend(["--queries"] + args.queries)
        if args.max_results:
            cmd_args.extend(["--max-results", str(args.max_results)])
        if args.output:
            cmd_args.extend(["--output", args.output])
        
        # Set PYTHONPATH to include src directory
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root / "src")
        
        # Run the phase1 script
        result = subprocess.run(cmd_args, cwd=str(project_root), env=env)
        
        return result.returncode
        
    except Exception as e:
        print(f"Error running Phase 1: {e}", file=sys.stderr)
        return 1


def cmd_literature(args) -> int:
    """Handle literature processing command."""
    try:
        from .phase1.literature_processor import LiteratureProcessor
        from .utils.config import load_config
        
        config = load_config(args.config)
        processor = LiteratureProcessor(config)
        
        documents = processor.process_query(
            query=args.query,
            max_results=args.max_results,
            output_file=args.output
        )
        
        print(f"Processed {len(documents)} documents")
        print(f"Results saved to {args.output}")
        return 0
        
    except Exception as e:
        print(f"Error processing literature: {e}", file=sys.stderr)
        return 1


def cmd_kg(args) -> int:
    """Handle knowledge graph processing command."""
    try:
        from .phase1.kg_preprocessor import KGPreprocessor
        from .utils.config import load_config
        
        config = load_config(args.config)
        processor = KGPreprocessor(config)
        
        # Download and process data
        processor.download_all_data()
        processor.process_all_data()
        processor.save_integrated_graph(args.output)
        
        stats = processor.graph_builder.get_statistics()
        print(f"Processed KG: {stats['num_entities']} entities, {stats['num_relations']} relations")
        print(f"Results saved to {args.output}")
        return 0
        
    except Exception as e:
        print(f"Error processing knowledge graphs: {e}", file=sys.stderr)
        return 1


def cmd_link(args) -> int:
    """Handle entity linking command."""
    try:
        from .phase1.literature_processor import LiteratureProcessor
        from .phase1.kg_preprocessor import KGPreprocessor
        from .phase1.entity_linker import EntityLinker
        from .utils.config import load_config
        
        config = load_config(args.config)
        
        # Load data
        lit_processor = LiteratureProcessor(config)
        kg_processor = KGPreprocessor(config)
        
        documents = lit_processor.load_results(args.literature)
        kg_processor.load_integrated_graph(args.kg)
        
        # Perform linking
        entity_linker = EntityLinker(config)
        entity_linker.load_kg_entities(kg_processor)
        
        results = entity_linker.batch_link_documents(documents)
        entity_linker.save_linking_results(results, args.output)
        
        total_links = sum(len(r.matches) for r in results)
        print(f"Created {total_links} entity links")
        print(f"Results saved to {args.output}")
        return 0
        
    except Exception as e:
        print(f"Error performing entity linking: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print("Configuration file not found. Run 'litkg setup' first.", file=sys.stderr)
        return 1
    
    # Dispatch to command handlers
    if args.command == "setup":
        return cmd_setup(args)
    elif args.command == "phase1":
        return cmd_phase1(args)
    elif args.command == "literature":
        return cmd_literature(args)
    elif args.command == "kg":
        return cmd_kg(args)
    elif args.command == "link":
        return cmd_link(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())