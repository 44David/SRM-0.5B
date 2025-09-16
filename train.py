import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

class SCoTDDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        ds = pd.read_json(path_or_buf=json_file, lines=True)
        for i in range(len(ds)):
            
            problem = ds['problem'][i]
            correct_answer = ds['correct_answer'][i]
            
            for trace in ds['thinking_traces'][i]:
                self.examples.append({
                    'problem': problem, 
                    'thinking': trace,
                    'answer': correct_answer
                })

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        
        text = f"Problem: {item['problem']}\n\nSolution:\n{item['thinking']}"
        
        tokens = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze()
        }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # tf32 (tensorfloat32) is a mode that accelerates fp32 convolutions and matmul
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul,'allow_tf32'):
            # allows cuda matmuls to use tf32
            torch.backends.cuda.matmul.allow_tf32 = True
        
        
    mixed_precision = device == 'cuda' and torch.cuda.get_device_capability()[0] >= 7
    
    #AMP - (Automatic Mixed Precision) GradScaler is used for efficient training by implementing loss scaling in gradient compute
    scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
    print(f"Use Mixed Precision: {mixed_precision}")
    
    train_log_format = {
        'epochs': [], 
        'batches': [], 
        'total_hours': 0, 
        'estimated_cost': 0, 
        'final_stats': {}
    }
    
    print("Loading Dataset")
    
    dataset = SCoTDDataset("SCoTD_math_reasoning.jsonl", tokenizer, max_length=256)
    
    print(f"Dataset size: {len(dataset)} examples loaded.")
    
    
    # this is achieved with gradient accumulation, 1. processes multiple batches, 2. accumulate gradients 3. then update gradients
    batch_size = 16
    gradient_accumulation = 4
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        # workers process input data batches, 
        # so there is always training data processed for the model to train. 4 means there are 4 processed batches ahead of model
        num_workers=4,
        pin_memory=True # locks RAM pages so they cant be swapped to disk
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    model.gradient_checkpointing_enable()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999))
    
    steps_per_epoch = len(dataset) // (batch_size * gradient_accumulation)
    num_epochs = 5
    total_steps = num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps) # t_max = epochs * steps_per_epoch
    
    
    model.train()
    training_logs = {
        'step_losses': [],
        'step_numbers': [],
        'epoch_losses': [],
        'learning_rates': [],
        'epoch_times': [],
        'total_steps': 0,
    }
    
    global_step = 0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_start_time = time.time()
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            labels = input_ids.clone()
            
            if mixed_precision and scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / gradient_accumulation
                    
                scaler.scale(loss).backward()
                
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation
                loss.backward()
                
            accumulated_loss += loss.item()
            
            
            if (batch_idx + 1) % gradient_accumulation == 0:
                
                global_step += 1
                
                if mixed_precision and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                scheduler.step()
                
                current_lr = scheduler.get_last_lr()[0]
                step_loss = accumulated_loss
                total_loss += step_loss * gradient_accumulation
                
                training_logs['step_losses'].append(step_loss)
                training_logs['step_numbers'].append(global_step)
                training_logs['learning_rates'].append(current_lr)
                
                
                optimizer.zero_grad()
                accumulated_loss = 0.0
                
                if global_step % 50 == 0:
                     print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Loss: {step_loss:.4f}, LR: {current_lr:.6f}")
                    
                    
        epoch_time = time.time() - epoch_start_time    
        steps_this_epoch = (len(train_loader) + gradient_accumulation - 1) // gradient_accumulation
        avg_epoch_loss = total_loss / steps_this_epoch
        
        training_logs['epoch_losses'].append(avg_epoch_loss)
        training_logs['epoch_times'].append(epoch_time)
        

        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_epoch_loss:.4f}")
        
        if (epoch+1) % 2 == 0:
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'training_logs': training_logs
            }, f'checkpoint_epoch_{epoch+1}.pt')
            
        torch.cuda.empty_cache()
    
    training_logs['total_steps'] = global_step
    with open('training_logs.json', 'w') as f:
        json.dump(training_logs, f, indent=2)
    
    print("Training complete")
        
    
main()