# inference.py
import torch
import argparse
from model.gpt import GPT, GPTConfig
from config import MODEL_CONFIG
import tiktoken

def load_model(checkpoint_path, device):
    config = GPTConfig(**MODEL_CONFIG)
    model = GPT(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def generate_text(model, prompt, max_length=50, device="cpu"):
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            _, next_token = torch.topk(probs, 1)
            tokens = torch.cat((tokens, next_token), dim=1)
    decoded = enc.decode(tokens[0].tolist())
    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Prompt text")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length for generation")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    generated = generate_text(model, args.prompt, args.max_length, device)
    print("Generated text:")
    print(generated)

if __name__ == "__main__":
    main()
