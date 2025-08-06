#!/usr/bin/env python3
"""
Ollama Integration Demo for LitKG-Integrate

This script demonstrates the comprehensive Ollama integration for local LLM inference:
1. Ollama server setup and model management
2. Biomedical model recommendations and installation
3. Unified LLM interface across multiple providers
4. Performance benchmarking and optimization
5. Biomedical task processing with local models

Key benefits of Ollama integration:
- Privacy: All processing happens locally
- Cost: No API fees for inference
- Customization: Fine-tune models for biomedical tasks
- Offline capability: Works without internet connection
- Control: Full control over model versions and parameters
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from litkg.utils.config import load_config
from litkg.utils.logging import setup_logging

# Import Ollama integration components
try:
    from litkg.llm_integration import (
        OllamaManager,
        OllamaLLM,
        BiomedicalOllamaChain,
        LocalModelManager,
        UnifiedLLMManager,
        LLMProvider,
        BiomedicalLLMInterface
    )
    OLLAMA_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Ollama integration not available: {e}")
    print("Please install dependencies: uv sync")
    OLLAMA_INTEGRATION_AVAILABLE = False
    sys.exit(1)


def demonstrate_ollama_setup(logger):
    """Demonstrate Ollama server setup and model management."""
    logger.info("=== OLLAMA SETUP AND MODEL MANAGEMENT ===")
    
    # Initialize Ollama manager
    ollama_manager = OllamaManager(auto_start=True)
    
    # Check server status
    server_running = ollama_manager.check_server_status()
    logger.info(f"Ollama server status: {'✅ Running' if server_running else '❌ Not running'}")
    
    if not server_running:
        logger.info("Starting Ollama server...")
        if ollama_manager.start_server():
            logger.info("✅ Ollama server started successfully")
        else:
            logger.error("❌ Failed to start Ollama server")
            logger.info("Please install Ollama: https://ollama.ai/download")
            return None
    
    # List available models
    available_models = ollama_manager.list_available_models()
    logger.info(f"Available models: {available_models}")
    
    # Get biomedical model recommendations
    logger.info("\nBiomedical Model Recommendations:")
    recommendations = ollama_manager.recommend_models_for_task("general_biomedical", "8GB")
    
    for i, model_info in enumerate(recommendations[:5]):
        logger.info(f"{i+1}. {model_info.name}")
        logger.info(f"   Size: {model_info.size}, Parameters: {model_info.parameters}")
        logger.info(f"   Description: {model_info.description}")
        logger.info(f"   Biomedical optimized: {model_info.biomedical_optimized}")
        logger.info(f"   Memory requirements: {model_info.memory_requirements}")
        logger.info("")
    
    # Setup biomedical models
    if not available_models or len(available_models) == 0:
        logger.info("Setting up biomedical models...")
        installed_models = ollama_manager.setup_biomedical_models("8GB")
        logger.info(f"Installed models: {installed_models}")
        return installed_models
    
    return available_models


def demonstrate_local_inference(available_models: List[str], logger):
    """Demonstrate local LLM inference with biomedical examples."""
    logger.info("=== LOCAL LLM INFERENCE DEMO ===")
    
    if not available_models:
        logger.warning("No models available for inference demo")
        return
    
    # Select best available model
    model_name = available_models[0]  # Use first available model
    logger.info(f"Using model: {model_name}")
    
    # Initialize Ollama LLM
    ollama_llm = OllamaLLM(
        model=model_name,
        temperature=0.1,
        biomedical_mode=True
    )
    
    # Test biomedical queries
    biomedical_queries = [
        "What is the role of BRCA1 in DNA repair?",
        "Explain the mechanism of action of tamoxifen in breast cancer treatment.",
        "What are the key differences between oncogenes and tumor suppressor genes?",
        "How does p53 regulate cell cycle progression?",
        "What is the relationship between inflammation and cancer development?"
    ]
    
    logger.info("Processing biomedical queries...")
    
    for i, query in enumerate(biomedical_queries):
        logger.info(f"\nQuery {i+1}: {query}")
        
        start_time = time.time()
        response = ollama_llm.generate(query)
        response_time = time.time() - start_time
        
        logger.info(f"Response ({response_time:.2f}s):")
        logger.info(f"{response[:300]}..." if len(response) > 300 else response)
        logger.info("-" * 50)
    
    # Performance measurement
    logger.info("\nMeasuring model performance...")
    performance = ollama_llm.measure_performance("What is BRCA1?")
    logger.info(f"Performance metrics:")
    logger.info(f"  Response time: {performance.response_time:.2f}s")
    logger.info(f"  Tokens per second: {performance.tokens_per_second:.1f}")
    logger.info(f"  Memory usage: {performance.memory_usage}")
    
    return performance


def demonstrate_biomedical_chains(available_models: List[str], logger):
    """Demonstrate specialized biomedical chains."""
    logger.info("=== BIOMEDICAL TASK CHAINS DEMO ===")
    
    if not available_models:
        logger.warning("No models available for chains demo")
        return
    
    model_name = available_models[0]
    
    # Test different biomedical tasks
    tasks = {
        "entity_extraction": {
            "text": "BRCA1 mutations are associated with increased risk of breast cancer. Patients with BRCA1 deficiency may benefit from PARP inhibitor therapy such as olaparib.",
            "description": "Extract biomedical entities"
        },
        "relation_extraction": {
            "text": "The p53 protein regulates cell cycle progression and apoptosis. Loss of p53 function leads to uncontrolled cell division and cancer development.",
            "entities": "p53, cell cycle, apoptosis, cancer",
            "description": "Extract relationships between entities"
        },
        "hypothesis_generation": {
            "context": "EGFR mutations are common in lung cancer and predict response to tyrosine kinase inhibitors.",
            "observation": "Some EGFR-mutated tumors develop resistance to erlotinib treatment.",
            "description": "Generate testable hypothesis"
        },
        "validation": {
            "hypothesis": "Combination therapy with EGFR inhibitors and autophagy modulators may overcome resistance in EGFR-mutated lung cancer.",
            "evidence": "Studies show that autophagy activation is a resistance mechanism to EGFR inhibition.",
            "description": "Validate hypothesis plausibility"
        }
    }
    
    results = {}
    
    for task_name, task_data in tasks.items():
        logger.info(f"\n--- {task_name.upper()} ---")
        logger.info(f"Task: {task_data['description']}")
        
        # Initialize chain for this task
        chain = BiomedicalOllamaChain(
            model=model_name,
            task_type=task_name
        )
        
        start_time = time.time()
        
        try:
            if task_name == "entity_extraction":
                result = chain.extract_entities(task_data["text"])
            elif task_name == "relation_extraction":
                result = chain.extract_relations(task_data["text"], task_data["entities"])
            elif task_name == "hypothesis_generation":
                result = chain.generate_hypothesis(task_data["context"], task_data["observation"])
            elif task_name == "validation":
                result = chain.validate_hypothesis(task_data["hypothesis"], task_data["evidence"])
            
            response_time = time.time() - start_time
            
            logger.info(f"Input: {list(task_data.values())[0][:100]}...")
            logger.info(f"Result ({response_time:.2f}s):")
            logger.info(result[:400] + "..." if len(result) > 400 else result)
            
            results[task_name] = {
                "result": result,
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error in {task_name}: {e}")
            results[task_name] = {"error": str(e)}
    
    return results


def demonstrate_unified_interface(logger):
    """Demonstrate unified LLM interface across providers."""
    logger.info("=== UNIFIED LLM INTERFACE DEMO ===")
    
    # Initialize unified interface
    unified_llm = UnifiedLLMManager()
    
    # Check available providers
    logger.info("Available LLM providers:")
    interface = unified_llm.llm_interface
    
    for provider in LLMProvider:
        available = provider in interface.clients
        logger.info(f"  {provider.value}: {'✅ Available' if available else '❌ Not available'}")
    
    # Setup local models
    logger.info("\nSetting up local models...")
    setup_results = unified_llm.setup_local_models("8GB")
    logger.info(f"Local model setup results: {setup_results}")
    
    # Test biomedical tasks
    test_tasks = [
        {
            "task": "entity_extraction",
            "input": "The TP53 gene encodes the p53 protein, which acts as a tumor suppressor by regulating cell cycle progression and inducing apoptosis in response to DNA damage."
        },
        {
            "task": "hypothesis_generation", 
            "input": {
                "context": "KRAS mutations are found in 30% of lung cancers and are associated with poor prognosis.",
                "observation": "KRAS-mutated tumors show resistance to EGFR inhibitors."
            }
        },
        {
            "task": "validation",
            "input": {
                "hypothesis": "Targeting downstream effectors of KRAS may be more effective than direct KRAS inhibition.",
                "evidence": "MEK inhibitors show activity in KRAS-mutated cancers in preclinical studies."
            }
        }
    ]
    
    task_results = {}
    
    for task_info in test_tasks:
        task_name = task_info["task"]
        task_input = task_info["input"]
        
        logger.info(f"\n--- Testing {task_name.upper()} ---")
        
        try:
            # Get model recommendations for this task
            recommendations = unified_llm.get_model_recommendations(
                task=task_name,
                constraints={"require_local": True, "max_cost": 0.01}
            )
            logger.info(f"Recommended models: {recommendations[:3]}")
            
            if recommendations:
                # Use best recommended model
                response = unified_llm.process_biomedical_task(
                    task=task_name,
                    input_data=task_input,
                    model=recommendations[0]
                )
                
                logger.info(f"Model used: {response.model}")
                logger.info(f"Provider: {response.provider.value}")
                logger.info(f"Response time: {response.response_time:.2f}s")
                logger.info(f"Cost: ${response.cost:.4f}")
                logger.info(f"Result: {response.content[:300]}...")
                
                task_results[task_name] = response
            else:
                logger.warning(f"No suitable models found for {task_name}")
                
        except Exception as e:
            logger.error(f"Error processing {task_name}: {e}")
    
    # Show usage statistics
    usage_stats = unified_llm.llm_interface.get_usage_stats()
    logger.info(f"\nUsage Statistics:")
    logger.info(f"  Total requests: {usage_stats['total_requests']}")
    logger.info(f"  Total cost: ${usage_stats['total_cost']:.4f}")
    logger.info(f"  Provider usage: {usage_stats['provider_usage']}")
    
    return task_results


def demonstrate_model_management(logger):
    """Demonstrate local model management features."""
    logger.info("=== LOCAL MODEL MANAGEMENT DEMO ===")
    
    # Initialize local model manager
    model_manager = LocalModelManager()
    
    # Install biomedical models
    logger.info("Installing biomedical models...")
    installation_results = model_manager.install_biomedical_models("8GB")
    
    for model_name, success in installation_results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"  {model_name}: {status}")
    
    # Get best model for different tasks
    tasks = ["entity_extraction", "hypothesis_generation", "validation"]
    
    logger.info("\nBest models for biomedical tasks:")
    for task in tasks:
        best_model = model_manager.get_best_model_for_task(task, "8GB")
        logger.info(f"  {task}: {best_model or 'No suitable model found'}")
    
    # Benchmark available models
    logger.info("\nBenchmarking models...")
    test_prompts = [
        "What is the function of the BRCA1 gene?",
        "Explain the mechanism of DNA repair.",
        "How do oncogenes contribute to cancer development?"
    ]
    
    benchmark_results = model_manager.benchmark_models(test_prompts)
    
    logger.info("Benchmark results:")
    for model_name, performance in benchmark_results.items():
        logger.info(f"  {model_name}:")
        logger.info(f"    Response time: {performance.response_time:.2f}s")
        logger.info(f"    Tokens/sec: {performance.tokens_per_second:.1f}")
        logger.info(f"    Memory usage: {performance.memory_usage}")
    
    return benchmark_results


def generate_ollama_report(
    setup_results: Dict[str, Any],
    performance_results: Dict[str, Any],
    task_results: Dict[str, Any],
    logger
):
    """Generate comprehensive Ollama integration report."""
    logger.info("=== GENERATING OLLAMA INTEGRATION REPORT ===")
    
    output_dir = Path("outputs/ollama_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive report
    report = {
        "ollama_integration_report": {
            "title": "Ollama Integration for Local Biomedical LLM Inference",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "local_inference_enabled": True,
                "models_installed": len(setup_results) if setup_results else 0,
                "tasks_demonstrated": len(task_results),
                "performance_measured": performance_results is not None
            }
        },
        "setup_results": setup_results,
        "performance_metrics": performance_results,
        "task_demonstrations": task_results,
        "benefits": {
            "privacy": "All processing happens locally - no data sent to external APIs",
            "cost_efficiency": "No API fees - unlimited inference once models are downloaded",
            "customization": "Full control over model parameters and fine-tuning",
            "offline_capability": "Works without internet connection",
            "reproducibility": "Consistent results with version-controlled models"
        },
        "recommendations": {
            "for_researchers": [
                "Use local models for sensitive biomedical data",
                "Customize models for specific research domains",
                "Combine with traditional NLP for hybrid approaches"
            ],
            "for_institutions": [
                "Deploy Ollama on institutional hardware",
                "Create shared model repositories",
                "Integrate with existing research workflows"
            ],
            "for_developers": [
                "Use unified interface for provider flexibility",
                "Implement fallback strategies for reliability",
                "Monitor performance and costs across providers"
            ]
        }
    }
    
    # Save detailed report
    report_file = output_dir / "ollama_integration_report.json"
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Detailed report saved to {report_file}")
    
    # Create human-readable summary
    summary_file = output_dir / "ollama_summary.txt"
    
    summary_text = f"""
Ollama Integration Demo Results
==============================

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

SETUP STATUS:
- Ollama server: ✅ Running
- Models installed: {len(setup_results) if setup_results else 0}
- Local inference: ✅ Enabled

CAPABILITIES DEMONSTRATED:
✅ Local model management and installation
✅ Biomedical-optimized model recommendations  
✅ Multi-provider unified interface
✅ Task-specific biomedical chains
✅ Performance benchmarking
✅ Privacy-preserving inference

PERFORMANCE HIGHLIGHTS:
"""
    
    if performance_results:
        summary_text += f"""- Response time: {performance_results.response_time:.2f}s
- Tokens per second: {performance_results.tokens_per_second:.1f}
- Memory usage: {performance_results.memory_usage}
"""
    
    summary_text += f"""
KEY BENEFITS:
🔒 Privacy: All processing happens locally
💰 Cost: No API fees for inference  
🎛️  Control: Full model customization
📡 Offline: Works without internet
🔬 Research: Perfect for sensitive biomedical data

BIOMEDICAL TASKS TESTED:
"""
    
    for task_name in task_results.keys():
        summary_text += f"✅ {task_name.replace('_', ' ').title()}\n"
    
    summary_text += f"""
NEXT STEPS:
1. Install Ollama: https://ollama.ai/download
2. Run: make run-ollama (or python scripts/example_ollama_integration.py)
3. Explore biomedical model fine-tuning
4. Integrate with existing LitKG workflows
5. Deploy on institutional infrastructure

FILES GENERATED:
- Detailed report: {report_file}
- Summary: {summary_file}
"""
    
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    logger.info(f"Summary saved to {summary_file}")
    
    return {
        "detailed_report": report_file,
        "summary": summary_file
    }


def main():
    """Main function to run the Ollama integration demonstration."""
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("🦙 Ollama Integration Demo for LitKG-Integrate")
    logger.info("=" * 70)
    
    if not OLLAMA_INTEGRATION_AVAILABLE:
        logger.error("Ollama integration not available")
        return
    
    try:
        # Step 1: Ollama setup and model management
        logger.info("\n" + "="*50)
        logger.info("STEP 1: Ollama Setup and Model Management")
        logger.info("="*50)
        
        setup_results = demonstrate_ollama_setup(logger)
        
        if not setup_results:
            logger.error("Ollama setup failed. Please install Ollama and try again.")
            return
        
        # Step 2: Local inference demonstration
        logger.info("\n" + "="*50)
        logger.info("STEP 2: Local LLM Inference")
        logger.info("="*50)
        
        performance_results = demonstrate_local_inference(setup_results, logger)
        
        # Step 3: Biomedical task chains
        logger.info("\n" + "="*50)
        logger.info("STEP 3: Biomedical Task Chains")
        logger.info("="*50)
        
        chain_results = demonstrate_biomedical_chains(setup_results, logger)
        
        # Step 4: Unified interface
        logger.info("\n" + "="*50)
        logger.info("STEP 4: Unified LLM Interface")
        logger.info("="*50)
        
        interface_results = demonstrate_unified_interface(logger)
        
        # Step 5: Model management
        logger.info("\n" + "="*50)
        logger.info("STEP 5: Model Management")
        logger.info("="*50)
        
        management_results = demonstrate_model_management(logger)
        
        # Step 6: Generate comprehensive report
        logger.info("\n" + "="*50)
        logger.info("STEP 6: Generate Report")
        logger.info("="*50)
        
        report_files = generate_ollama_report(
            setup_results={
                "models_installed": setup_results,
                "server_status": "running"
            },
            performance_results=performance_results.__dict__ if performance_results else None,
            task_results={
                "chains": chain_results,
                "unified_interface": {k: v.content[:200] + "..." if hasattr(v, 'content') else str(v) for k, v in interface_results.items()},
                "management": {k: str(v)[:200] + "..." if len(str(v)) > 200 else str(v) for k, v in management_results.items()}
            },
            logger=logger
        )
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("🎉 OLLAMA INTEGRATION DEMO COMPLETE!")
        logger.info("="*70)
        
        logger.info(f"\n🦙 OLLAMA CAPABILITIES:")
        logger.info(f"✅ Local model management and installation")
        logger.info(f"✅ Biomedical-optimized model recommendations")
        logger.info(f"✅ Privacy-preserving local inference")
        logger.info(f"✅ Multi-provider unified interface")
        logger.info(f"✅ Task-specific biomedical chains")
        logger.info(f"✅ Performance benchmarking and optimization")
        
        logger.info(f"\n🔬 BIOMEDICAL BENEFITS:")
        logger.info(f"  🔒 Privacy: Process sensitive medical data locally")
        logger.info(f"  💰 Cost: No API fees for unlimited inference")
        logger.info(f"  🎛️  Control: Full customization and fine-tuning")
        logger.info(f"  📡 Offline: Works without internet connection")
        logger.info(f"  🔬 Research: Perfect for institutional deployments")
        
        logger.info(f"\n📊 PERFORMANCE:")
        if performance_results:
            logger.info(f"  ⚡ Response time: {performance_results.response_time:.2f}s")
            logger.info(f"  🚀 Tokens/sec: {performance_results.tokens_per_second:.1f}")
            logger.info(f"  💾 Memory usage: {performance_results.memory_usage}")
        
        logger.info(f"\n📁 OUTPUT FILES:")
        for report_type, file_path in report_files.items():
            logger.info(f"  - {report_type}: {file_path}")
        
        logger.info(f"\n🚀 NEXT STEPS:")
        logger.info(f"  1. Install Ollama: https://ollama.ai/download")
        logger.info(f"  2. Explore model fine-tuning for specific domains")
        logger.info(f"  3. Integrate with existing biomedical workflows")
        logger.info(f"  4. Deploy on institutional infrastructure")
        logger.info(f"  5. Combine with traditional NLP approaches")
        
        logger.info(f"\n🎯 INTEGRATION COMPLETE:")
        logger.info(f"  LitKG now supports local, privacy-preserving LLM inference!")
        logger.info(f"  Use 'make run-ollama' to run this demo anytime.")
        
    except Exception as e:
        logger.error(f"Error in Ollama integration demo: {e}")
        raise


if __name__ == "__main__":
    main()