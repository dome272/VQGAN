import argparse
import torch
from PIL import Image
from torchvision import transforms
from vqgan import VQGAN
import math

def load_model(args):
    model = VQGAN(args)
    if torch.cuda.is_available():
        checkpoint = torch.load(args.checkpoint_path)
    else:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model.eval()
    return model

def process_image(image_path, image_size, device):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def inference(model, image):
    with torch.no_grad():
        codebook_mapping, codebook_indices, _ = model.encode(image)
        
        # Print shapes for debugging
        print("codebook_mapping shape:", codebook_mapping.shape)
        print("codebook_indices shape:", codebook_indices.shape)
        
        # Decode using codebook_mapping
        reconstructed_image = model.decode(codebook_mapping)
        
    return reconstructed_image, codebook_indices

def save_image(image, path):
    transform = transforms.ToPILImage()
    image = transform(image.squeeze(0).cpu().clamp(-1, 1) * 0.5 + 0.5)
    image.save(path)

def main(args):
    model = load_model(args)
    input_image = process_image(args.input_image, args.image_size, args.device)
    
    reconstructed_image, tokens = inference(model, input_image)
    
    save_image(reconstructed_image, args.output_image)
    print("Imagen reconstruida guardada en:", args.output_image)
    print("Tokens discretos:", tokens.cpu().flatten().tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN Inference")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Which device to use')
    parser.add_argument('--checkpoint_path', type=str, default='/teamspace/studios/this_studio/checkpoints/vqganima_epoch_1.pt', help='Path to the trained model checkpoint')
    parser.add_argument('--input_image', type=str, default='/teamspace/studios/this_studio/1.jpeg', help='Path to the input image')
    parser.add_argument('--output_image', type=str, default='/teamspace/studios/this_studio/imag.png', help='Path to save the reconstructed image')
    
    args = parser.parse_args()
    main(args)
