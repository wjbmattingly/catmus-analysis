import pandas as pd
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import pickle
import tqdm

class HandwrittenTextDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_data = self.df['im'][idx]
        text = self.df['text'][idx]
        language = self.df['language'][idx]
        shelfmark = self.df['shelfmark'][idx]
        project = self.df['project'][idx]

        # Ensure image_data is in numpy array format
        if isinstance(image_data, Image.Image):
            image = image_data
        else:
            image = Image.fromarray(image_data)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, padding="max_length", 
                                          max_length=self.max_target_length, truncation=True).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        # Convert image to numpy array if not already
        if not isinstance(image_data, np.ndarray):
            image_data = np.array(image)

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
            "text": text,
            "original_image": image_data,
            "language": language,
            "index": idx,
            "project": project,
            "shelfmark": shelfmark
        }
def load_and_prepare_dataset(script="Cursiva", lang=None, split="test"):
    dataset = load_dataset("CATMuS/medieval", split=split)
    filtered_dataset = dataset.filter(lambda example: example['script_type'] == script)
    if lang:
        filtered_dataset = filtered_dataset.filter(lambda example: example['language'] == lang)
    return pd.DataFrame(filtered_dataset)

def compute_cer_per_image(model, processor, dataset, device, cer_threshold):
    cer_metric = load_metric("cer")
    model.to(device)
    model.eval()
    flagged_results = []

    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
        pixel_values = data["pixel_values"].unsqueeze(0).to(device)
        outputs = model.generate(pixel_values)
        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)

        labels = data["labels"].unsqueeze(0)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        if cer > cer_threshold:
            flagged_results.append({
                "original_image": data["original_image"],
                "text": data["text"],
                "predicted_text": pred_str[0],
                "cer": cer,
                "shelfmark": data["shelfmark"],
                "project": data["project"],
                "language": data["language"]
            })

    return pd.DataFrame(flagged_results)

def save_flagged_results(df, filename):
    # Ensure serialization of numpy arrays (use pickle if necessary)
    df['original_image'] = df['original_image'].apply(lambda x: pickle.dumps(x) if not isinstance(x, bytes) else x)
    df.to_parquet(f"{filename}.parquet", index=False)

def process_dataset(script, lang, split, from_pretrained_model, device, cer_threshold, output_filename, **kwargs):
    df = load_and_prepare_dataset(script=script, lang=lang, split=split)
    processor = TrOCRProcessor.from_pretrained(from_pretrained_model)
    dataset = HandwrittenTextDataset(df, processor)
    model = VisionEncoderDecoderModel.from_pretrained(from_pretrained_model)
    
    flagged_df = compute_cer_per_image(model, processor, dataset, device, cer_threshold)
    save_flagged_results(flagged_df, output_filename)
    print(f"Flagged results saved to {output_filename}")

# Example usage
process_dataset(
    script="Caroline", # Make sure the Script aligns with the model
    lang='Latin', # Make sure the language aligns with the model. Use None to use all languages
    split='validation', 
    from_pretrained_model="./Caroline", # Change this to the HugggingFace repo
    device='mps', 
    cer_threshold=0.0, 
    output_filename="flagged_caroline_latin_validation"
)
