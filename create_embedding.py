#!/usr/bin/env python3
"""
VoiceForge Embedding File Creator

Simple utility to create embedding files for testing and development.
"""

import argparse
import json
import numpy as np
import os
from pathlib import Path


def create_random_embedding(output_path, shape=(1, 192), seed=None):
    """Create a random embedding file for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    embedding = np.random.randn(*shape)
    
    if output_path.endswith('.npy'):
        np.save(output_path, embedding)
        print(f"✅ Created NumPy embedding: {output_path}")
    elif output_path.endswith('.json'):
        data = {
            'spk_emb': embedding.tolist(),
            'spk_emb_shape': list(embedding.shape),
            'embedding_type': 'random_test',
            'embedding_dim': embedding.shape[-1],
            'metadata': 'Random embedding for testing'
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Created JSON embedding: {output_path}")
    else:
        raise ValueError("Output file must end with .npy or .json")


def create_from_vector(output_path, vector_string, shape=(1, 192)):
    """Create embedding file from comma-separated vector string."""
    try:
        # Parse comma-separated values
        values = [float(x.strip()) for x in vector_string.split(',')]
        
        if len(values) != shape[1]:
            print(f"⚠️  Warning: Expected {shape[1]} values, got {len(values)}")
            # Pad or truncate to match expected shape
            if len(values) < shape[1]:
                values.extend([0.0] * (shape[1] - len(values)))
            else:
                values = values[:shape[1]]
        
        embedding = np.array(values).reshape(shape)
        
        if output_path.endswith('.npy'):
            np.save(output_path, embedding)
            print(f"✅ Created NumPy embedding: {output_path}")
        elif output_path.endswith('.json'):
            data = {
                'spk_emb': embedding.tolist(),
                'spk_emb_shape': list(embedding.shape),
                'embedding_type': 'from_vector',
                'embedding_dim': embedding.shape[-1],
                'metadata': 'Embedding created from vector input'
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Created JSON embedding: {output_path}")
        else:
            raise ValueError("Output file must end with .npy or .json")
            
    except ValueError as e:
        print(f"❌ Error parsing vector: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Create embedding files for VoiceForge")
    parser.add_argument('output_path', help='Output file path (.npy or .json)')
    parser.add_argument('--random', action='store_true', help='Create random embedding')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--vector', type=str, help='Comma-separated vector values')
    parser.add_argument('--shape', type=str, default='1,192', help='Embedding shape (default: 1,192)')
    
    args = parser.parse_args()
    
    # Parse shape
    try:
        shape = tuple(int(x) for x in args.shape.split(','))
    except ValueError:
        print("❌ Invalid shape format. Use comma-separated integers (e.g., 1,192)")
        return
    
    # Ensure output directory exists
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.random:
        create_random_embedding(args.output_path, shape, args.seed)
    elif args.vector:
        create_from_vector(args.output_path, args.vector, shape)
    else:
        print("❌ Please specify either --random or --vector")
        parser.print_help()


if __name__ == "__main__":
    main() 