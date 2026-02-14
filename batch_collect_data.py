"""
batch_collect_data.py

Batch data collection for multiple ASL signs.
Iterates through a predefined list of signs and collects data for each.
"""

from collect_data import collect_data_for_sign
import json


VOCABULARY = [
    "hello",
    "thank_you",
    "please",
    "sorry",
    "yes",
    "no",
    "help",
    "love",
    "family",
    "friend",
    "eat",
    "drink",
    "sleep",
    "work",
    "school",
    "home",
    "go",
    "stop",
    "want",
    "need",
]


def batch_collect_data(
    vocabulary: list[str],
    sequences_per_sign: int = 30,
    output_dir: str = 'training_data'
):
    """
    Collect data for multiple signs in batch.
    
    Args:
        vocabulary: List of sign labels to collect
        sequences_per_sign: Number of sequences per sign
        output_dir: Directory to save data
    """
    print("\n" + "="*70)
    print("BATCH ASL DATA COLLECTION")
    print("="*70)
    print(f"\nTotal signs to collect: {len(vocabulary)}")
    print(f"Sequences per sign: {sequences_per_sign}")
    print(f"Total sequences: {len(vocabulary) * sequences_per_sign}")
    print(f"\nVocabulary: {', '.join(vocabulary[:10])}...")
    print("="*70 + "\n")
    
    response = input("Ready to start? This will take a while. (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled by user.")
        return
    
    for sign_index, sign_label in enumerate(vocabulary):
        print(f"\n\n{'='*70}")
        print(f"  SIGN {sign_index + 1}/{len(vocabulary)}: {sign_label.upper()}")
        print(f"{'='*70}")
        
        skip = input(f"Collect data for '{sign_label}'? (yes/skip/quit): ").lower()
        
        if skip == 'quit':
            print("\nBatch collection stopped by user.")
            break
        elif skip == 'skip':
            print(f"Skipping '{sign_label}'...")
            continue
        
        collect_data_for_sign(
            sign_label=sign_label,
            sign_index=sign_index,
            num_sequences=sequences_per_sign,
            output_dir=output_dir
        )
    
    print("\n" + "="*70)
    print("BATCH COLLECTION COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print("1. Review collected data in '{output_dir}' directory")
    print("2. Run: python train_model.py")
    print("3. Run: streamlit run app.py")
    
    labels_dict = {i: label for i, label in enumerate(vocabulary)}
    with open('vocabulary.json', 'w') as f:
        json.dump(labels_dict, f, indent=2)
    print(f"\n✓ Vocabulary saved to 'vocabulary.json'")


def continue_previous_collection(output_dir: str = 'training_data'):
    """
    Continue a previously interrupted batch collection.
    Detects which signs have already been collected.
    """
    import os
    
    print("\nScanning existing data...")
    
    collected_signs = set()
    if os.path.exists(output_dir):
        for dirname in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, dirname)):
                parts = dirname.split('_', 1)
                if len(parts) == 2:
                    collected_signs.add(parts[1])
    
    print(f"Found {len(collected_signs)} already collected signs:")
    print(f"  {', '.join(sorted(collected_signs))}")
    
    remaining = [sign for sign in VOCABULARY if sign not in collected_signs]
    
    if not remaining:
        print("\n✓ All signs in vocabulary have been collected!")
        return
    
    print(f"\nRemaining signs to collect: {len(remaining)}")
    print(f"  {', '.join(remaining[:10])}...")
    
    start_index = len(VOCABULARY) - len(remaining)
    for i, sign_label in enumerate(remaining):
        sign_index = start_index + i
        
        skip = input(f"\nCollect '{sign_label}'? (yes/skip/quit): ").lower()
        if skip == 'quit':
            break
        elif skip == 'skip':
            continue
        
        collect_data_for_sign(
            sign_label=sign_label,
            sign_index=sign_index,
            num_sequences=30,
            output_dir=output_dir
        )


def main():
    """Interactive menu for batch collection."""
    print("\n" + "="*70)
    print("ASL BATCH DATA COLLECTION TOOL")
    print("="*70)
    
    print("\nOptions:")
    print("1. Start new batch collection")
    print("2. Continue previous collection")
    print("3. Customize vocabulary and collect")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        batch_collect_data(VOCABULARY)
    elif choice == '2':
        continue_previous_collection()
    elif choice == '3':
        print("\nEnter signs separated by commas:")
        custom_input = input("Signs: ")
        custom_vocab = [s.strip().lower().replace(' ', '_') for s in custom_input.split(',')]
        batch_collect_data(custom_vocab)
    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
