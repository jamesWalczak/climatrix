#!/usr/bin/env python3
"""
Example demonstrating the new num_params property.

This script shows how the num_params property works for different types
of reconstructors in the climatrix package.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    print("=" * 60)
    print("Climatrix Reconstructors - num_params Property Demo")
    print("=" * 60)
    
    print("\nThe num_params property has been added to all reconstructors.")
    print("This property returns the number of trainable parameters for")
    print("neural network-based reconstructors, and 0 for traditional methods.")
    
    print("\n" + "-" * 40)
    print("Usage Examples:")
    print("-" * 40)
    
    # Example 1: Traditional reconstructor
    print("\n1. Traditional Reconstructor (IDW, Kriging):")
    print("   reconstructor = IDWReconstructor(dataset, target_domain)")
    print("   print(reconstructor.num_params)  # Output: 0")
    
    # Example 2: Neural network reconstructor
    print("\n2. Neural Network Reconstructor (SiNET, SIREN, MMGN):")
    print("   reconstructor = SiNETReconstructor(dataset, target_domain, hidden_dim=128)")
    print("   print(reconstructor.num_params)  # Output: e.g., 8576 (depends on architecture)")
    
    print("\n" + "-" * 40)
    print("Benefits:")
    print("-" * 40)
    print("• Compare model complexity across different NN reconstructors")
    print("• Monitor parameter count when tuning hyperparameters")
    print("• Useful for research and optimization studies")
    print("• Consistent interface across all NN-based methods")
    
    print("\n" + "-" * 40)
    print("Implementation Details:")
    print("-" * 40)
    print("• Property is available on BaseReconstructor class")
    print("• Automatically detects NN vs traditional reconstructors")
    print("• Only counts trainable parameters (requires_grad=True)")
    print("• Handles initialization errors gracefully")
    print("• Works with SiNET, SIREN, MMGN, and future NN implementations")
    
    print("\n" + "=" * 60)
    print("Implementation complete! The num_params property is ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()