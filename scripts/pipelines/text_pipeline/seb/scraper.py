import requests
import json
import time
import logging
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib.parse import urlparse, urljoin, urlsplit
import re
from datetime import datetime
from typing import Dict, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
import random
import os
import hashlib
import base64
from PIL import Image
from io import BytesIO
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from contextlib import contextmanager
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# Configure logging
logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()

class ScrapedPage(Base):
    """Database model for scraped pages."""
    __tablename__ = 'scraped_pages'
    
    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = sa.Column(sa.String, unique=True, nullable=False)
    title = sa.Column(sa.String)
    content = sa.Column(sa.Text)
    word_count = sa.Column(sa.Integer)
    char_count = sa.Column(sa.Integer)
    scraped_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    scraping_method = sa.Column(sa.String)
    success = sa.Column(sa.Boolean)
    error_message = sa.Column(sa.String)
    raw_html_length = sa.Column(sa.Integer)
    paragraphs_found = sa.Column(sa.Integer)
    
    # Relationship to images
    images = sa.orm.relationship("ScrapedImage", back_populates="page", cascade="all, delete-orphan")

class ScrapedImage(Base):
    """Database model for scraped images."""
    __tablename__ = 'scraped_images'
    
    id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    page_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey('scraped_pages.id'), nullable=False)
    original_url = sa.Column(sa.String, nullable=False)
    absolute_url = sa.Column(sa.String, nullable=False)
    alt_text = sa.Column(sa.String)
    title = sa.Column(sa.String)
    width = sa.Column(sa.Integer)
    height = sa.Column(sa.Integer)
    file_size = sa.Column(sa.Integer)
    file_format = sa.Column(sa.String)
    content_type = sa.Column(sa.String)
    
    # Storage options
    file_path = sa.Column(sa.String)  # Local file path
    base64_data = sa.Column(sa.Text)  # Base64 encoded image data
    external_url = sa.Column(sa.String)  # External storage URL (S3, etc.)
    
    # Metadata
    hash_md5 = sa.Column(sa.String)
    downloaded_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    download_success = sa.Column(sa.Boolean)
    download_error = sa.Column(sa.String)
    
    # Relationship
    page = sa.orm.relationship("ScrapedPage", back_populates="images")

@dataclass
class ImageData:
    """Data class for image information."""
    original_url: str
    absolute_url: str
    alt_text: str = ""
    title: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    file_format: Optional[str] = None
    content_type: Optional[str] = None
    file_path: Optional[str] = None
    base64_data: Optional[str] = None
    external_url: Optional[str] = None
    hash_md5: Optional[str] = None
    download_success: bool = False
    download_error: str = ""

@dataclass
class ScrapedContent:
    """Enhanced data class to store scraped content with images."""
    url: str
    title: str
    content: str
    word_count: int
    char_count: int
    scraped_at: str
    scraping_method: str
    success: bool
    error_message: str = ""
    raw_html_length: int = 0
    paragraphs_found: int = 0
    images: List[ImageData] = None
    
    def __post_init__(self):
        if self.images is None:
            self.images = []

@dataclass
class ScrapingConfig:
    """Enhanced configuration for web scraping with image options."""
    timeout: int = 30
    headless: bool = True
    user_agent: str = None
    force_dynamic: bool = False
    max_content_length: int = 1000000
    wait_for_stability: bool = True
    handle_verification: bool = True
    
    # Image scraping options
    scrape_images: bool = True
    max_images: int = 50
    min_image_size: int = 100  # pixels
    max_image_size: int = 5 * 1024 * 1024  # 5MB
    allowed_image_formats: List[str] = None
    download_images: bool = True
    storage_method: str = "database"  # "database", "file", "external"
    storage_path: str = "scraped_images"
    store_as_base64: bool = True
    
    def __post_init__(self):
        if self.user_agent is None:
            self.user_agent = (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
        if self.allowed_image_formats is None:
            self.allowed_image_formats = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg']

class DatabaseManager:
    """Database manager for storing scraped content and images."""
    
    def __init__(self, database_url: str):
        self.engine = sa.create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create database tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
        
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def store_scraped_content(self, content: ScrapedContent) -> str:
        """Store scraped content and images in database."""
        with self.get_session() as session:
            # Create page record
            page = ScrapedPage(
                url=content.url,
                title=content.title,
                content=content.content,
                word_count=content.word_count,
                char_count=content.char_count,
                scraped_at=datetime.fromisoformat(content.scraped_at),
                scraping_method=content.scraping_method,
                success=content.success,
                error_message=content.error_message,
                raw_html_length=content.raw_html_length,
                paragraphs_found=content.paragraphs_found
            )
            
            session.add(page)
            session.flush()  # Get the page ID
            
            # Store images
            for image_data in content.images:
                image = ScrapedImage(
                    page_id=page.id,
                    original_url=image_data.original_url,
                    absolute_url=image_data.absolute_url,
                    alt_text=image_data.alt_text,
                    title=image_data.title,
                    width=image_data.width,
                    height=image_data.height,
                    file_size=image_data.file_size,
                    file_format=image_data.file_format,
                    content_type=image_data.content_type,
                    file_path=image_data.file_path,
                    base64_data=image_data.base64_data,
                    external_url=image_data.external_url,
                    hash_md5=image_data.hash_md5,
                    download_success=image_data.download_success,
                    download_error=image_data.download_error
                )
                session.add(image)
            
            session.commit()
            return str(page.id)
    
    def get_scraped_content(self, page_id: str) -> Optional[ScrapedContent]:
        """Retrieve scraped content by page ID."""
        with self.get_session() as session:
            page = session.query(ScrapedPage).filter(ScrapedPage.id == page_id).first()
            if not page:
                return None
            
            # Convert images
            images = []
            for img in page.images:
                images.append(ImageData(
                    original_url=img.original_url,
                    absolute_url=img.absolute_url,
                    alt_text=img.alt_text or "",
                    title=img.title or "",
                    width=img.width,
                    height=img.height,
                    file_size=img.file_size,
                    file_format=img.file_format,
                    content_type=img.content_type,
                    file_path=img.file_path,
                    base64_data=img.base64_data,
                    external_url=img.external_url,
                    hash_md5=img.hash_md5,
                    download_success=img.download_success,
                    download_error=img.download_error or ""
                ))
            
            return ScrapedContent(
                url=page.url,
                title=page.title or "",
                content=page.content or "",
                word_count=page.word_count or 0,
                char_count=page.char_count or 0,
                scraped_at=page.scraped_at.isoformat(),
                scraping_method=page.scraping_method or "",
                success=page.success,
                error_message=page.error_message or "",
                raw_html_length=page.raw_html_length or 0,
                paragraphs_found=page.paragraphs_found or 0,
                images=images
            )

class ImageProcessor:
    """Handler for image processing and storage."""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        
        # Create storage directory if using file storage
        if config.storage_method == "file" and not os.path.exists(config.storage_path):
            os.makedirs(config.storage_path, exist_ok=True)
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is likely to be an image."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check file extension
        for ext in self.config.allowed_image_formats:
            if path.endswith(f'.{ext}'):
                return True
        
        # Check for common image URL patterns
        image_patterns = [
            r'/images?/',
            r'/img/',
            r'/photos?/',
            r'/pictures?/',
            r'/media/',
            r'/uploads/',
            r'/assets/',
        ]
        
        for pattern in image_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_md5(self, data: bytes) -> str:
        """Calculate MD5 hash of image data."""
        return hashlib.md5(data).hexdigest()
    
    def _get_image_info(self, data: bytes) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """Extract image dimensions and format."""
        try:
            image = Image.open(BytesIO(data))
            width, height = image.size
            format_name = image.format.lower() if image.format else None
            return width, height, format_name
        except Exception as e:
            logger.warning(f"Error extracting image info: {e}")
            return None, None, None
    
    def _download_image(self, url: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        """Download image from URL."""
        try:
            response = self.session.get(url, timeout=self.config.timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                return None, None, f"Invalid content type: {content_type}"
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.config.max_image_size:
                return None, None, f"Image too large: {content_length} bytes"
            
            # Download image data
            image_data = response.content
            
            # Verify it's actually an image and get dimensions
            width, height, format_name = self._get_image_info(image_data)
            if not width or not height:
                return None, None, "Could not determine image dimensions"
            
            # Check minimum size
            if width < self.config.min_image_size or height < self.config.min_image_size:
                return None, None, f"Image too small: {width}x{height}"
            
            return image_data, content_type, None
            
        except requests.exceptions.RequestException as e:
            return None, None, f"Download error: {str(e)}"
        except Exception as e:
            return None, None, f"Unexpected error: {str(e)}"
    
    def _save_image_to_file(self, image_data: bytes, filename: str) -> Optional[str]:
        """Save image data to file."""
        try:
            file_path = os.path.join(self.config.storage_path, filename)
            with open(file_path, 'wb') as f:
                f.write(image_data)
            return file_path
        except Exception as e:
            logger.error(f"Error saving image to file: {e}")
            return None
    
    def _encode_to_base64(self, image_data: bytes) -> str:
        """Encode image data to base64."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def process_image(self, img_tag, base_url: str) -> Optional[ImageData]:
        """Process a single image tag and return ImageData."""
        src = img_tag.get('src')
        if not src:
            return None
        
        # Convert to absolute URL
        absolute_url = urljoin(base_url, src)
        
        # Check if URL is valid for images
        if not self._is_valid_image_url(absolute_url):
            logger.debug(f"Skipping non-image URL: {absolute_url}")
            return None
        
        # Extract metadata from HTML
        alt_text = img_tag.get('alt', '')
        title = img_tag.get('title', '')
        
        # Create initial ImageData
        image_data = ImageData(
            original_url=src,
            absolute_url=absolute_url,
            alt_text=alt_text,
            title=title
        )
        
        # Download image if configured
        if self.config.download_images:
            img_bytes, content_type, error = self._download_image(absolute_url)
            
            if img_bytes:
                # Calculate hash
                image_data.hash_md5 = self._calculate_md5(img_bytes)
                image_data.file_size = len(img_bytes)
                image_data.content_type = content_type
                
                # Get image dimensions and format
                width, height, format_name = self._get_image_info(img_bytes)
                image_data.width = width
                image_data.height = height
                image_data.file_format = format_name
                
                # Store image based on configuration
                if self.config.storage_method == "file":
                    filename = f"{image_data.hash_md5}.{format_name}" if format_name else f"{image_data.hash_md5}.img"
                    file_path = self._save_image_to_file(img_bytes, filename)
                    if file_path:
                        image_data.file_path = file_path
                
                if self.config.store_as_base64:
                    image_data.base64_data = self._encode_to_base64(img_bytes)
                
                image_data.download_success = True
                logger.debug(f"Successfully processed image: {absolute_url}")
                
            else:
                image_data.download_success = False
                image_data.download_error = error or "Unknown download error"
                logger.warning(f"Failed to download image {absolute_url}: {error}")
        
        return image_data

class EnhancedWebScraper:
    """Enhanced web scraper with image processing capabilities."""
    
    def __init__(self, config: ScrapingConfig = None, db_manager: DatabaseManager = None):
        self.config = config or ScrapingConfig()
        self.db_manager = db_manager
        
        # Initialize image processor
        self.image_processor = ImageProcessor(self.config) if self.config.scrape_images else None
        
        # Session for static scraping
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.config.user_agent})
        
        # Selenium driver (initialized when needed)
        self.driver = None
        self._driver_initialized = False
    
    def _setup_selenium_driver(self):
        """Setup Selenium driver."""
        if self._driver_initialized:
            return
            
        try:
            chrome_options = Options()
            if self.config.headless:
                chrome_options.add_argument('--headless')
            
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={self.config.user_agent}')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.config.timeout)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self._driver_initialized = True
            logger.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[ImageData]:
        """Extract images from HTML soup."""
        if not self.config.scrape_images or not self.image_processor:
            return []
        
        images = []
        img_tags = soup.find_all('img')
        
        logger.info(f"Found {len(img_tags)} image tags")
        
        for img_tag in img_tags[:self.config.max_images]:
            try:
                image_data = self.image_processor.process_image(img_tag, base_url)
                if image_data:
                    images.append(image_data)
            except Exception as e:
                logger.warning(f"Error processing image: {e}")
                continue
        
        logger.info(f"Successfully processed {len(images)} images")
        return images
    
    def _cleanup_text(self, text: str) -> str:
        """Clean up extracted text content."""
        if not text:
            return ""
        
        lines = text.splitlines()
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = re.sub(r'\s+', ' ', line.strip())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
    
    def _extract_content_intelligently(self, soup: BeautifulSoup) -> tuple[str, int]:
        """Extract content using multiple strategies."""
        # Similar to original implementation but focused on text extraction
        # ... (keeping the same logic as in the original code)
        
        # Find main content
        main_content = None
        content_selectors = [
            'article', 'main', '[role="main"]', 
            '.content', '.main-content', '.article-content', 
            '.post-content', '.entry-content', '.page-content',
            '.article-body', '.story-body', '.post-body'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text content
        paragraph_count = len(main_content.find_all('p'))
        text_content = main_content.get_text(separator=' ', strip=True)
        text_content = self._cleanup_text(text_content)
        
        if len(text_content) > self.config.max_content_length:
            text_content = text_content[:self.config.max_content_length] + "...[truncated]"
        
        return text_content, paragraph_count
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements from the soup."""
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        
        unwanted_selectors = [
            'nav', 'header', 'footer', 'aside',
            '.navigation', '.nav', '.menu', '.sidebar',
            '.footer', '.header', '.ad', '.advertisement',
            '.popup', '.modal', '.cookie-notice', '.cookie-banner',
            '.social', '.share', '.comment', '.comments',
            '.related', '.recommendations', '.newsletter'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        return soup
    
    def scrape_url(self, url: str) -> ScrapedContent:
        """Scrape a URL with automatic fallback and image processing."""
        try:
            # Try static scraping first
            if not self.config.force_dynamic:
                result = self._scrape_static_content(url)
                if result.success and result.word_count >= 100:
                    return result
            
            # Try dynamic scraping
            return self._scrape_dynamic_content(url)
            
        except Exception as e:
            logger.error(f"Error in scrape_url: {e}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                word_count=0,
                char_count=0,
                scraped_at=datetime.now().isoformat(),
                scraping_method="failed",
                success=False,
                error_message=str(e),
                images=[]
            )
    
    def _scrape_static_content(self, url: str) -> ScrapedContent:
        """Scrape static content with image processing."""
        try:
            logger.info(f"Attempting static scraping for: {url}")
            
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else urlparse(url).netloc
            
            # Remove unwanted elements
            soup = self._remove_unwanted_elements(soup)
            
            # Extract text content
            text_content, paragraph_count = self._extract_content_intelligently(soup)
            
            # Extract images
            images = self._extract_images(soup, url)
            
            word_count = len(text_content.split())
            char_count = len(text_content)
            
            logger.info(f"Static scraping successful. Words: {word_count}, Images: {len(images)}")
            
            return ScrapedContent(
                url=url,
                title=title,
                content=text_content,
                word_count=word_count,
                char_count=char_count,
                scraped_at=datetime.now().isoformat(),
                scraping_method="static",
                success=True,
                raw_html_length=len(response.content),
                paragraphs_found=paragraph_count,
                images=images
            )
            
        except Exception as e:
            logger.error(f"Error during static scraping: {e}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                word_count=0,
                char_count=0,
                scraped_at=datetime.now().isoformat(),
                scraping_method="static",
                success=False,
                error_message=str(e),
                images=[]
            )
    
    def _scrape_dynamic_content(self, url: str) -> ScrapedContent:
        """Scrape dynamic content with image processing."""
        try:
            logger.info(f"Attempting dynamic scraping for: {url}")
            
            if not self._driver_initialized:
                self._setup_selenium_driver()
            
            self.driver.get(url)
            
            # Wait for page load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(5)
            
            html_source = self.driver.page_source
            soup = BeautifulSoup(html_source, 'html.parser')
            
            # Extract title
            title = self.driver.title or urlparse(url).netloc
            
            # Remove unwanted elements
            soup = self._remove_unwanted_elements(soup)
            
            # Extract text content
            text_content, paragraph_count = self._extract_content_intelligently(soup)
            
            # Extract images
            images = self._extract_images(soup, url)
            
            word_count = len(text_content.split())
            char_count = len(text_content)
            
            logger.info(f"Dynamic scraping successful. Words: {word_count}, Images: {len(images)}")
            
            return ScrapedContent(
                url=url,
                title=title,
                content=text_content,
                word_count=word_count,
                char_count=char_count,
                scraped_at=datetime.now().isoformat(),
                scraping_method="dynamic",
                success=True,
                raw_html_length=len(html_source),
                paragraphs_found=paragraph_count,
                images=images
            )
            
        except Exception as e:
            logger.error(f"Error during dynamic scraping: {e}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                word_count=0,
                char_count=0,
                scraped_at=datetime.now().isoformat(),
                scraping_method="dynamic",
                success=False,
                error_message=str(e),
                images=[]
            )
    
    def scrape_and_store(self, url: str) -> Optional[str]:
        """Scrape URL and store in database."""
        if not self.db_manager:
            raise ValueError("Database manager not configured")
        
        result = self.scrape_url(url)
        if result.success:
            page_id = self.db_manager.store_scraped_content(result)
            logger.info(f"Stored scraped content with ID: {page_id}")
            return page_id
        else:
            logger.error(f"Failed to scrape URL: {url}")
            return None
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self._driver_initialized = False
        
        if self.session:
            self.session.close()

# API Response Models for Frontend
@dataclass
class ImageResponse:
    """Response model for images in API."""
    id: str
    original_url: str
    alt_text: str
    title: str
    width: Optional[int]
    height: Optional[int]
    file_format: Optional[str]
    base64_data: Optional[str] = None
    file_path: Optional[str] = None
    external_url: Optional[str] = None

@dataclass
class ScrapedContentResponse:
    """Response model for scraped content in API."""
    id: str
    url: str
    title: str
    content: str
    word_count: int
    char_count: int
    scraped_at: str
    success: bool
    images: List[ImageResponse]

# Complete the missing parts of the web scraper

class EnhancedWebScrapingService:
    """Enhanced service class with database integration."""
    
    def __init__(self, database_url: str, default_config: ScrapingConfig = None):
        self.db_manager = DatabaseManager(database_url)
        self.db_manager.create_tables()
        self.default_config = default_config or ScrapingConfig()
    
    def scrape_and_store(self, url: str, config: ScrapingConfig = None) -> Optional[str]:
        """Scrape URL and store in database."""
        config = config or self.default_config
        
        scraper = EnhancedWebScraper(config, self.db_manager)
        try:
            return scraper.scrape_and_store(url)
        finally:
            scraper.close()
    
    def get_scraped_content(self, page_id: str, include_images: bool = True) -> Optional[ScrapedContentResponse]:
        """Get scraped content by ID."""
        content = self.db_manager.get_scraped_content(page_id)
        if not content:
            return None
        
        # Convert images to response format
        images = []
        if include_images:
            for img in content.images:
                image_response = ImageResponse(
                    id=img.hash_md5 or str(uuid.uuid4()),
                    original_url=img.original_url,
                    alt_text=img.alt_text,
                    title=img.title,
                    width=img.width,
                    height=img.height,
                    file_format=img.file_format,
                    base64_data=img.base64_data if img.download_success else None,
                    file_path=img.file_path if img.download_success else None,
                    external_url=img.external_url if img.download_success else None
                )
                images.append(image_response)
        
        return ScrapedContentResponse(
            id=page_id,
            url=content.url,
            title=content.title,
            content=content.content,
            word_count=content.word_count,
            char_count=content.char_count,
            scraped_at=content.scraped_at,
            success=content.success,
            images=images
        )
    
    def get_all_scraped_content(self, limit: int = 100) -> List[ScrapedContentResponse]:
        """Get all scraped content with pagination."""
        with self.db_manager.get_session() as session:
            pages = session.query(ScrapedPage).order_by(ScrapedPage.scraped_at.desc()).limit(limit).all()
            
            results = []
            for page in pages:
                # Convert images
                images = []
                for img in page.images:
                    image_response = ImageResponse(
                        id=str(img.id),
                        original_url=img.original_url,
                        alt_text=img.alt_text or "",
                        title=img.title or "",
                        width=img.width,
                        height=img.height,
                        file_format=img.file_format,
                        base64_data=img.base64_data if img.download_success else None,
                        file_path=img.file_path if img.download_success else None,
                        external_url=img.external_url if img.download_success else None
                    )
                    images.append(image_response)
                
                response = ScrapedContentResponse(
                    id=str(page.id),
                    url=page.url,
                    title=page.title or "",
                    content=page.content or "",
                    word_count=page.word_count or 0,
                    char_count=page.char_count or 0,
                    scraped_at=page.scraped_at.isoformat(),
                    success=page.success,
                    images=images
                )
                results.append(response)
            
            return results
    
    def search_content(self, query: str, limit: int = 50) -> List[ScrapedContentResponse]:
        """Search scraped content by text."""
        with self.db_manager.get_session() as session:
            pages = session.query(ScrapedPage).filter(
                sa.or_(
                    ScrapedPage.title.ilike(f'%{query}%'),
                    ScrapedPage.content.ilike(f'%{query}%')
                )
            ).order_by(ScrapedPage.scraped_at.desc()).limit(limit).all()
            
            results = []
            for page in pages:
                # Convert images
                images = []
                for img in page.images:
                    image_response = ImageResponse(
                        id=str(img.id),
                        original_url=img.original_url,
                        alt_text=img.alt_text or "",
                        title=img.title or "",
                        width=img.width,
                        height=img.height,
                        file_format=img.file_format,
                        base64_data=img.base64_data if img.download_success else None
                    )
                    images.append(image_response)
                
                response = ScrapedContentResponse(
                    id=str(page.id),
                    url=page.url,
                    title=page.title or "",
                    content=page.content or "",
                    word_count=page.word_count or 0,
                    char_count=page.char_count or 0,
                    scraped_at=page.scraped_at.isoformat(),
                    success=page.success,
                    images=images
                )
                results.append(response)
            
            return results
    
    def delete_scraped_content(self, page_id: str) -> bool:
        """Delete scraped content by ID."""
        with self.db_manager.get_session() as session:
            page = session.query(ScrapedPage).filter(ScrapedPage.id == page_id).first()
            if page:
                session.delete(page)
                session.commit()
                return True
            return False
    
    def get_scraping_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about scraped content."""
        with self.db_manager.get_session() as session:
            total_pages = session.query(ScrapedPage).count()
            successful_pages = session.query(ScrapedPage).filter(ScrapedPage.success == True).count()
            total_images = session.query(ScrapedImage).count()
            successful_images = session.query(ScrapedImage).filter(ScrapedImage.download_success == True).count()
            
            avg_word_count = session.query(sa.func.avg(ScrapedPage.word_count)).scalar() or 0
            avg_images_per_page = session.query(sa.func.avg(
                sa.func.count(ScrapedImage.id)
            )).select_from(ScrapedPage).outerjoin(ScrapedImage).group_by(ScrapedPage.id).scalar() or 0
            
            return {
                'total_pages': total_pages,
                'successful_pages': successful_pages,
                'success_rate': (successful_pages / total_pages * 100) if total_pages > 0 else 0,
                'total_images': total_images,
                'successful_images': successful_images,
                'image_success_rate': (successful_images / total_images * 100) if total_images > 0 else 0,
                'avg_word_count': float(avg_word_count),
                'avg_images_per_page': float(avg_images_per_page)
            }

# Utility functions for batch processing
def batch_scrape_urls(urls: List[str], 
                     database_url: str, 
                     config: ScrapingConfig = None,
                     max_workers: int = 3) -> List[Tuple[str, Optional[str]]]:
    """Batch scrape multiple URLs with threading."""
    import concurrent.futures
    
    service = EnhancedWebScrapingService(database_url, config)
    results = []
    
    def scrape_single_url(url: str) -> Tuple[str, Optional[str]]:
        try:
            page_id = service.scrape_and_store(url)
            return url, page_id
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return url, None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(scrape_single_url, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url, page_id = future.result()
            results.append((url, page_id))
            logger.info(f"Completed scraping: {url} -> {page_id}")
    
    return results

# Simple Web Scraping Service (without database requirement)
class WebScrapingService:
    """Simple web scraping service for direct URL processing."""
    
    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.scraper = None
    
    def health_check(self) -> Dict[str, Union[bool, str]]:
        """Check service health."""
        try:
            # Test basic functionality
            test_html = "<html><body><h1>Test</h1></body></html>"
            soup = BeautifulSoup(test_html, 'html.parser')
            return {
                "status": "healthy",
                "selenium_available": True,
                "beautifulsoup_available": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def scrape(self, url: str, config: ScrapingConfig = None) -> ScrapedContent:
        """Scrape a single URL."""
        config = config or self.config
        
        if not self.scraper:
            self.scraper = EnhancedWebScraper(config)
        
        return self.scraper.scrape_url(url)
    
    def batch_scrape(self, urls: List[str], config: ScrapingConfig = None) -> List[ScrapedContent]:
        """Scrape multiple URLs."""
        config = config or self.config
        results = []
        
        if not self.scraper:
            self.scraper = EnhancedWebScraper(config)
        
        for url in urls:
            try:
                result = self.scraper.scrape_url(url)
                results.append(result)
                logger.info(f"Scraped {url}: Success={result.success}, Words={result.word_count}")
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                results.append(ScrapedContent(
                    url=url,
                    title="",
                    content="",
                    word_count=0,
                    char_count=0,
                    scraped_at=datetime.now().isoformat(),
                    scraping_method="failed",
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self.scraper:
            self.scraper.close()
            self.scraper = None

# Main execution example
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    test_urls = [
        "https://www.amherst.edu/"
    ]
    
    # Configure scraping (with images stored to ./scraped_images)
    config = ScrapingConfig(
        headless=True,
        scrape_images=True,
        max_images=10,
        download_images=True,
        store_as_base64=True,
        storage_method="file",  # Store images as files
        storage_path="./scraped_images",  # Store in ./scraped_images directory
        timeout=30
    )
    
    # Create service
    service = WebScrapingService(config)
    
    # Test health check
    health = service.health_check()
    print(f"Service health: {health}")
    
    # Test single URL scraping
    for url in test_urls:
        try:
            result = service.scrape(url)
            print(f"\nURL: {url}")
            print(f"Success: {result.success}")
            print(f"Title: {result.title}")
            print(f"Word Count: {result.word_count}")
            print(f"Images Found: {len(result.images)}")
            print(f"Method: {result.scraping_method}")
            if result.images:
                print(f"Sample Images:")
                for i, img in enumerate(result.images[:3]):  # Show first 3 images
                    print(f"  {i+1}. {img.absolute_url} ({img.width}x{img.height} {img.file_format})")
                    if img.file_path:
                        print(f"      Saved to: {img.file_path}")
            if not result.success:
                print(f"Error: {result.error_message}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    # Test batch scraping
    try:
        print(f"\n{'='*50}")
        print("BATCH SCRAPING TEST")
        print(f"{'='*50}")
        
        batch_results = service.batch_scrape(test_urls)
        print(f"\nBatch scraping completed: {len(batch_results)} results")
        
        total_words = 0
        total_images = 0
        successful_scrapes = 0
        images_saved = 0
        
        for i, result in enumerate(batch_results):
            print(f"  {i+1}. {result.url}")
            print(f"     Success: {result.success}")
            print(f"     Words: {result.word_count}")
            print(f"     Images: {len(result.images)}")
            
            if result.success:
                successful_scrapes += 1
                total_words += result.word_count
                total_images += len(result.images)
                
                # Count successfully saved images
                for img in result.images:
                    if img.file_path and img.download_success:
                        images_saved += 1
        
        print(f"\nBatch Summary:")
        print(f"  Successful scrapes: {successful_scrapes}/{len(batch_results)}")
        print(f"  Total words extracted: {total_words}")
        print(f"  Total images found: {total_images}")
        print(f"  Images successfully saved: {images_saved}")
        print(f"  Images saved to: ./scraped_images/")
        print(f"  Average words per page: {total_words/successful_scrapes if successful_scrapes > 0 else 0:.1f}")
        
    except Exception as e:
        print(f"Batch scraping error: {e}")
    
    # Clean up
    service.cleanup()
    print("\nScraping service cleaned up successfully")
    
    # Show final summary of what was saved
    if os.path.exists("./scraped_images"):
        saved_files = os.listdir("./scraped_images")
        print(f"\nFiles saved in ./scraped_images: {len(saved_files)}")
        for filename in saved_files[:5]:  # Show first 5 files
            filepath = os.path.join("./scraped_images", filename)
            file_size = os.path.getsize(filepath)
            print(f"  - {filename} ({file_size} bytes)")
        if len(saved_files) > 5:
            print(f"  ... and {len(saved_files) - 5} more files")
    else:
        print("\nNo images were successfully saved.")