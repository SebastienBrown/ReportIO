import os
import requests
from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image
from google_images_search import GoogleImagesSearch
import numpy as np
from urllib.parse import urlparse
import hashlib

# Try different CLIP import methods
CLIP_TYPE = None

try:
    import clip
    if hasattr(clip, "load"):  # only OpenAI's clip has this
        CLIP_TYPE = "openai_clip"
    else:
        raise ImportError
except ImportError:
    print("Standard clip import failed or doesn't have 'load'. Trying OpenCLIP...")
    try:
        import open_clip as clip
        CLIP_TYPE = "open_clip"
    except ImportError:
        print("OpenCLIP not found either. Will use HuggingFace transformers.")
        CLIP_TYPE = "huggingface"

class ImageSearchPipeline:
    def __init__(self, google_api_key: str, google_cse_id: str):
        """
        Initialize the image search pipeline.
        
        Args:
            google_api_key: Google Custom Search API key
            google_cse_id: Google Custom Search Engine ID
        """
        self.gis = GoogleImagesSearch(google_api_key, google_cse_id)
        self.clip_type = CLIP_TYPE
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEBUG] clip_type: {self.clip_type}")
        
        try:
            if self.clip_type == "openai_clip":
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            elif self.clip_type == "open_clip":
                self.clip_model, _, self.clip_preprocess = clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
                self.clip_model = self.clip_model.to(self.device)
                self.clip_tokenizer = clip.get_tokenizer('ViT-B-32')
            else:  # huggingface
                from transformers import CLIPProcessor, CLIPModel
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = self.clip_model.to(self.device)
                self.clip_type = "huggingface"
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            print("Trying alternative CLIP loading method...")
            try:
                # Alternative method - direct from huggingface
                from transformers import CLIPProcessor, CLIPModel
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = self.clip_model.to(self.device)
                self.clip_type = "huggingface"
            except Exception as e2:
                print(f"Alternative CLIP loading also failed: {e2}")
                raise
        
        print(f"Using device: {self.device}")
        print(f"CLIP model loaded successfully using {self.clip_type}")
    
    def search_images(self, query: str, num_images: int = 20) -> List[dict]:
        """
        Search for images using Google Images Search.
        
        Args:
            query: Search query
            num_images: Number of images to retrieve
            
        Returns:
            List of image metadata dictionaries
        """
        print(f"Searching for '{query}' - retrieving {num_images} images...")
        
        search_params = {
            'q': query,
            'num': num_images,
            'fileType': 'jpg,png,jpeg',
            'safe': 'active',
            'imgSize': 'MEDIUM'
        }
        
        try:
            self.gis.search(search_params=search_params)
            
            images_data = []
            for image in self.gis.results():
                # Get available attributes from the GSImage object
                image_info = {
                    'url': image.url,
                    'referrer_url': getattr(image, 'referrer_url', ''),
                    'size': getattr(image, 'size', {}),
                    'format': getattr(image, 'format', ''),
                    'width': getattr(image, 'width', 0),
                    'height': getattr(image, 'height', 0)
                }
                
                # Add optional attributes if they exist
                if hasattr(image, 'title'):
                    image_info['title'] = image.title
                if hasattr(image, 'context'):
                    image_info['context'] = image.context
                if hasattr(image, 'thumbnail_url'):
                    image_info['thumbnail_url'] = image.thumbnail_url
                
                images_data.append(image_info)
            
            print(f"Found {len(images_data)} images")
            return images_data
            
        except Exception as e:
            print(f"Error during image search: {e}")
            print("Trying with simplified search parameters...")
            
            # Fallback with minimal parameters
            simplified_params = {
                'q': query,
                'num': min(num_images, 10),
                'safe': 'active'
            }
            
            try:
                self.gis.search(search_params=simplified_params)
                
                images_data = []
                for image in self.gis.results():
                    images_data.append({
                        'url': image.url,
                        'referrer_url': getattr(image, 'referrer_url', ''),
                        'size': getattr(image, 'size', {}),
                        'format': getattr(image, 'format', ''),
                    })
                
                print(f"Found {len(images_data)} images with simplified search")
                return images_data
                
            except Exception as e2:
                print(f"Fallback search also failed: {e2}")
                return []
    
    def download_image(self, url: str, save_path: str) -> bool:
        """
        Download an image from URL.
        
        Args:
            url: Image URL
            save_path: Local path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # Verify the image can be opened
            Image.open(save_path).verify()
            return True
            
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False
    
    def calculate_clip_similarity(self, image_path: str, text_prompt: str) -> float:
        """
        Calculate CLIP similarity between an image and text prompt.
        
        Args:
            image_path: Path to the image
            text_prompt: Text prompt to compare against
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            if self.clip_type == "openai_clip":
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                text_tokens = clip.tokenize([text_prompt]).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_tensor)
                    text_features = self.clip_model.encode_text(text_tokens)
                    
                    # Normalize features
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate cosine similarity
                    similarity = torch.cosine_similarity(image_features, text_features).item()
            
            elif self.clip_type == "open_clip":
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                text_tokens = self.clip_tokenizer([text_prompt]).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_tensor)
                    text_features = self.clip_model.encode_text(text_tokens)
                    
                    # Normalize features
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate cosine similarity
                    similarity = torch.cosine_similarity(image_features, text_features).item()
            
            else:  # huggingface
                inputs = self.clip_preprocess(text=[text_prompt], images=image, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    similarity = torch.cosine_similarity(
                        outputs.image_embeds, 
                        outputs.text_embeds
                    ).item()
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity for {image_path}: {e}")
            return 0.0
    
    def create_filename(self, url: str, index: int) -> str:
        """
        Create a unique filename for the image.
        
        Args:
            url: Image URL
            index: Index in the list
            
        Returns:
            Filename string
        """
        # Get file extension from URL
        parsed_url = urlparse(url)
        path = parsed_url.path
        ext = Path(path).suffix
        
        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            ext = '.jpg'
        
        # Create unique filename using URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"image_{index:03d}_{url_hash}{ext}"
        
        return filename
    
    def run_pipeline(self, 
                    query: str, 
                    similarity_prompt: str = None,
                    initial_count: int = 10,
                    shortlist_count: int = 10,
                    final_count: int = 5,
                    output_dir: str = "downloaded_images") -> List[Tuple[str, float]]:
        """
        Run the complete image search similarity pipeline.
        
        Args:
            query: Search query for Google Images
            similarity_prompt: Text prompt for CLIP similarity (defaults to query)
            initial_count: Number of images to initially retrieve
            shortlist_count: Number of images to download for similarity testing
            final_count: Number of final images to keep
            output_dir: Directory to save images
            
        Returns:
            List of tuples (image_path, similarity_score)
        """
        if similarity_prompt is None:
            similarity_prompt = query
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        temp_path = output_path / "temp"
        temp_path.mkdir(exist_ok=True)
        
        print(f"Starting pipeline for query: '{query}'")
        print(f"Similarity prompt: '{similarity_prompt}'")
        print(f"Output directory: {output_path}")
        
        # Step 1: Search for images
        images_data = self.search_images(query, initial_count)
        
        if not images_data:
            print("No images found!")
            return []
        
        # Step 2: Download shortlist of images
        print(f"\nDownloading shortlist of {min(shortlist_count, len(images_data))} images...")
        downloaded_images = []
        
        for i, img_data in enumerate(images_data[:shortlist_count]):
            filename = self.create_filename(img_data['url'], i)
            temp_image_path = temp_path / filename
            
            if self.download_image(img_data['url'], str(temp_image_path)):
                downloaded_images.append({
                    'path': str(temp_image_path),
                    'data': img_data,
                    'index': i
                })
                print(f"Downloaded: {filename}")
            else:
                print(f"Failed to download image {i}")
        
        print(f"Successfully downloaded {len(downloaded_images)} images")
        
        # Step 3: Calculate CLIP similarities
        print(f"\nCalculating CLIP similarities...")
        similarities = []
        
        for img_info in downloaded_images:
            similarity = self.calculate_clip_similarity(img_info['path'], similarity_prompt)
            similarities.append({
                'path': img_info['path'],
                'similarity': similarity,
                'data': img_info['data'],
                'index': img_info['index']
            })
            print(f"Image {img_info['index']}: similarity = {similarity:.4f}")
        
        # Step 4: Sort by similarity and select top images
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_images = similarities[:final_count]
        
        print(f"\nTop {final_count} most similar images:")
        final_results = []
        
        for i, img_info in enumerate(top_images):
            # Move image to final directory
            temp_path_obj = Path(img_info['path'])
            final_filename = f"final_{i+1:02d}_{temp_path_obj.name}"
            final_path = output_path / final_filename
            
            temp_path_obj.rename(final_path)
            
            final_results.append((str(final_path), img_info['similarity']))
            print(f"  {i+1}. {final_filename} - similarity: {img_info['similarity']:.4f}")
        
        # Clean up temporary files
        for img_info in similarities:
            temp_file = Path(img_info['path'])
            if temp_file.exists():
                temp_file.unlink()
        
        temp_path.rmdir()
        
        print(f"\nPipeline completed! Final images saved to: {output_path}")
        return final_results

# Example usage
def main():
    # You need to set these environment variables or replace with your actual keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'your_google_api_key_here')
    GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID', 'your_google_cse_id_here')
    
    if GOOGLE_API_KEY == 'your_google_api_key_here' or GOOGLE_CSE_ID == 'your_google_cse_id_here':
        print("Please set your Google API key and Custom Search Engine ID!")
        print("You can get these from:")
        print("1. Google Cloud Console: https://console.cloud.google.com/")
        print("2. Google Custom Search Engine: https://cse.google.com/")
        return
    
    # Initialize pipeline
    pipeline = ImageSearchPipeline(GOOGLE_API_KEY, GOOGLE_CSE_ID)
    
    # Run the pipeline
    query = "desert dunes under a starry night sky"
    similarity_prompt = query
    
    results = pipeline.run_pipeline(
        query=query,
        similarity_prompt=similarity_prompt,
        initial_count=10,
        shortlist_count=10,
        final_count=5,
        output_dir="CLIP_output"
    )
    
    print(f"\nFinal results:")
    for image_path, similarity in results:
        print(f"{image_path}: {similarity:.4f}")

if __name__ == "__main__":
    main()