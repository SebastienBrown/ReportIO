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
from urllib.parse import urlparse, urljoin
import re
from datetime import datetime
from typing import Dict, Optional, List, Union
from dataclasses import dataclass, asdict
import random
import os
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    """Data class to store scraped content."""
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

@dataclass
class ScrapingConfig:
    """Configuration for web scraping."""
    timeout: int = 30
    headless: bool = True
    user_agent: str = None
    force_dynamic: bool = False
    max_content_length: int = 1000000  # 1MB limit
    wait_for_stability: bool = True
    handle_verification: bool = True
    
    def __post_init__(self):
        if self.user_agent is None:
            self.user_agent = (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )

class WebScrapingError(Exception):
    """Custom exception for web scraping errors."""
    pass

class WebScraper:
    """Universal web scraper that handles both static and dynamic content."""
    
    def __init__(self, config: ScrapingConfig = None):
        """
        Initialize the web scraper.
        
        Args:
            config: ScrapingConfig object with scraping parameters
        """
        self.config = config or ScrapingConfig()
        
        # Session for static scraping
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.config.user_agent})
        
        # Selenium driver (initialized when needed)
        self.driver = None
        self._driver_initialized = False
    
    def _setup_selenium_driver(self):
        """Enhanced Selenium setup with better anti-detection."""
        if self._driver_initialized:
            return
            
        try:
            chrome_options = Options()
            if self.config.headless:
                chrome_options.add_argument('--headless')
            
            # Enhanced anti-detection options
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={self.config.user_agent}')
            
            # Additional anti-detection measures
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.config.timeout)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self._driver_initialized = True
            logger.info("Enhanced Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise WebScrapingError(f"Failed to initialize browser: {e}")
    
    def _cleanup_text(self, text: str) -> str:
        """Clean up extracted text content while preserving paragraph structure."""
        if not text:
            return ""
        
        # Split into lines and clean each line
        lines = text.splitlines()
        cleaned_lines = []
        
        for line in lines:
            # Clean each line but preserve structure
            cleaned_line = re.sub(r'\s+', ' ', line.strip())
            if cleaned_line:  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)
        
        # Join lines with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive newlines (more than 2)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
    
    def _extract_content_intelligently(self, soup: BeautifulSoup) -> tuple[str, int]:
        """
        Extract content using multiple strategies to capture all text content.
        
        Returns:
            tuple: (extracted_text, paragraph_count)
        """
        # Strategy 1: Look for main content containers first
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
                logger.debug(f"Found main content using selector: {selector}")
                break
        
        # Strategy 2: If no main content found, look for the element with most text content
        if not main_content:
            candidates = soup.find_all(['div', 'section', 'article'])
            best_candidate = None
            max_content_score = 0
            
            for candidate in candidates:
                # Skip if it's likely navigation or unwanted content
                if candidate.get('class'):
                    class_str = ' '.join(candidate.get('class', [])).lower()
                    if any(unwanted in class_str for unwanted in ['nav', 'menu', 'sidebar', 'footer', 'header', 'ad']):
                        continue
                
                # Score based on paragraphs + other text content
                paragraph_count = len(candidate.find_all('p'))
                text_length = len(candidate.get_text(strip=True))
                content_score = paragraph_count * 10 + text_length / 100
                
                if content_score > max_content_score:
                    max_content_score = content_score
                    best_candidate = candidate
            
            if best_candidate and max_content_score > 0:
                main_content = best_candidate
                logger.debug(f"Found main content with score {max_content_score:.1f}")
        
        # Strategy 3: Fall back to body if nothing else works
        if not main_content:
            main_content = soup.find('body')
            logger.debug("Using body as main content")
        
        if not main_content:
            main_content = soup
            logger.debug("Using entire soup as main content")
        
        # Extract text content comprehensively
        paragraph_count = len(main_content.find_all('p'))
        
        # Method 1: Extract by element type with hierarchy preservation
        content_elements = []
        
        # Get all text-containing elements in document order
        for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'blockquote', 'li', 'td', 'th', 'span', 'div']):
            # Skip if this element contains other text elements (to avoid duplication)
            child_text_elements = elem.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'blockquote', 'li'])
            if child_text_elements:
                continue
            
            elem_text = elem.get_text(strip=True)
            if elem_text and len(elem_text) > 5:  # Include shorter text for headings, etc.
                # Check for duplicate content (common in nested structures)
                if not any(elem_text in existing for existing in content_elements):
                    content_elements.append(elem_text)
        
        # Method 2: If Method 1 doesn't capture enough, use comprehensive text extraction
        if len(content_elements) < 3 or sum(len(elem) for elem in content_elements) < 500:
            logger.debug("Using comprehensive text extraction method")
            
            # Get all text nodes, preserving structure
            text_parts = []
            
            # Process different element types with appropriate formatting
            for elem in main_content.descendants:
                if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 2:
                        text_parts.append(f"\n{text}\n")
                elif elem.name == 'p':
                    text = elem.get_text(strip=True)
                    if text and len(text) > 10:
                        text_parts.append(text)
                elif elem.name in ['li', 'td', 'th']:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 5:
                        text_parts.append(text)
                elif elem.name == 'br':
                    text_parts.append('\n')
                elif elem.name is None:  # Text node
                    text = str(elem).strip()
                    if text and len(text) > 2 and not text.isspace():
                        # Only add if it's not already covered by parent elements
                        if not any(text in part for part in text_parts[-3:]):
                            text_parts.append(text)
            
            # Clean and join
            cleaned_parts = []
            for part in text_parts:
                if part.strip():
                    cleaned_parts.append(part.strip())
            
            extracted_text = ' '.join(cleaned_parts)
            extracted_text = self._cleanup_text(extracted_text)
            
        else:
            # Use Method 1 results
            extracted_text = '\n\n'.join(content_elements)
            extracted_text = self._cleanup_text(extracted_text)
        
        # Method 3: Final fallback - extract everything and clean aggressively
        if not extracted_text or len(extracted_text.split()) < 50:
            logger.debug("Using fallback full text extraction")
            extracted_text = main_content.get_text(separator=' ', strip=True)
            extracted_text = self._cleanup_text(extracted_text)
        
        # Truncate if too long
        if len(extracted_text) > self.config.max_content_length:
            extracted_text = extracted_text[:self.config.max_content_length] + "...[truncated]"
        
        return extracted_text, paragraph_count
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements from the soup."""
        # Remove script and style tags
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        
        # Remove common unwanted elements
        unwanted_selectors = [
            'nav', 'header', 'footer', 'aside',
            '.navigation', '.nav', '.menu', '.sidebar',
            '.footer', '.header', '.ad', '.advertisement',
            '.popup', '.modal', '.cookie-notice', '.cookie-banner',
            '.social', '.share', '.comment', '.comments',
            '.related', '.recommendations', '.newsletter',
            '[class*="ad-"]', '[class*="ads-"]', '[id*="ad-"]',
            '[class*="cookie"]', '[class*="popup"]', '[class*="modal"]'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove elements with low text content but high link density
        for div in soup.find_all('div'):
            text = div.get_text(strip=True)
            links = div.find_all('a')
            if text and len(text) < 100 and len(links) > 3:
                div.decompose()
        
        return soup
    
    def _wait_for_content_stability(self, max_wait: int = 30) -> bool:
        """
        Wait for page content to stabilize (stop changing).
        
        Args:
            max_wait: Maximum time to wait in seconds
            
        Returns:
            bool: True if content stabilized, False if timeout
        """
        if not self.config.wait_for_stability:
            return True
            
        logger.debug("Waiting for content to stabilize...")
        
        previous_content_hash = None
        stable_count = 0
        
        for i in range(max_wait // 2):  # Check every 2 seconds
            try:
                current_content = self.driver.page_source
                current_hash = hash(current_content)
                
                if current_hash == previous_content_hash:
                    stable_count += 1
                    if stable_count >= 3:  # Content unchanged for 6 seconds
                        logger.debug("Content appears stable")
                        return True
                else:
                    stable_count = 0
                    previous_content_hash = current_hash
                
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error checking content stability: {e}")
                time.sleep(2)
        
        logger.debug("Content stability check timeout")
        return False
    
    def _wait_for_captcha_resolution(self, max_wait_time: int = 60) -> bool:
        """
        Wait for CAPTCHA or security verification to complete.
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            bool: True if verification appears to be complete, False if timeout
        """
        if not self.config.handle_verification:
            return True
            
        logger.info("Detected security verification, waiting for resolution...")
        
        # Common indicators that verification is in progress
        verification_indicators = [
            "verifying you are human",
            "security check",
            "checking your browser",
            "please wait",
            "verification in progress",
            "cloudflare",
            "ddos protection",
            "just a moment"
        ]
        
        start_time = time.time()
        last_content = ""
        
        while time.time() - start_time < max_wait_time:
            try:
                # Get current page content
                current_content = self.driver.page_source.lower()
                
                # Check if we're still seeing verification indicators
                verification_active = any(indicator in current_content for indicator in verification_indicators)
                
                if not verification_active:
                    logger.info("Security verification appears to be complete")
                    return True
                
                # Check if content has changed (indication of progress)
                if current_content != last_content:
                    logger.debug("Page content changed, verification may be progressing...")
                    last_content = current_content
                
                # Wait a bit before checking again
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error while waiting for verification: {e}")
                time.sleep(2)
        
        logger.warning(f"Verification wait timeout after {max_wait_time} seconds")
        return False

    def _scrape_static_content(self, url: str) -> ScrapedContent:
        """
        Scrape static HTML content using requests and BeautifulSoup.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedContent object with scraped data
        """
        try:
            logger.info(f"Attempting static scraping for: {url}")
            
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            raw_html_length = len(response.content)
            logger.debug(f"Retrieved HTML content: {raw_html_length} bytes")
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else urlparse(url).netloc
            
            # Debug: Count paragraphs before cleaning
            initial_p_count = len(soup.find_all('p'))
            logger.debug(f"Found {initial_p_count} paragraph tags in raw HTML")
            
            # Remove unwanted elements
            soup = self._remove_unwanted_elements(soup)
            
            # Debug: Count paragraphs after cleaning
            cleaned_p_count = len(soup.find_all('p'))
            logger.debug(f"Paragraphs remaining after cleaning: {cleaned_p_count}")
            
            # Extract content intelligently
            text_content, paragraph_count = self._extract_content_intelligently(soup)
            
            word_count = len(text_content.split())
            char_count = len(text_content)
            
            logger.info(f"Static scraping successful. Words: {word_count}, Characters: {char_count}, Paragraphs: {paragraph_count}")
            
            return ScrapedContent(
                url=url,
                title=title,
                content=text_content,
                word_count=word_count,
                char_count=char_count,
                scraped_at=datetime.now().isoformat(),
                scraping_method="static",
                success=True,
                raw_html_length=raw_html_length,
                paragraphs_found=paragraph_count
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during static scraping: {e}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                word_count=0,
                char_count=0,
                scraped_at=datetime.now().isoformat(),
                scraping_method="static",
                success=False,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error during static scraping: {e}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                word_count=0,
                char_count=0,
                scraped_at=datetime.now().isoformat(),
                scraping_method="static",
                success=False,
                error_message=str(e)
            )
    
    def _scrape_dynamic_content(self, url: str) -> ScrapedContent:
        """
        Enhanced dynamic scraping with CAPTCHA handling.
        """
        try:
            logger.info(f"Attempting dynamic scraping for: {url}")
            
            # Initialize Selenium driver if not already done
            if not self._driver_initialized:
                self._setup_selenium_driver()
            
            # Load the page
            self.driver.get(url)
            
            # Wait for initial page load
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.warning("Initial page load timeout, continuing...")
            
            # Check if we're facing a security verification
            page_content = self.driver.page_source.lower()
            verification_keywords = [
                "verifying you are human",
                "security check",
                "checking your browser",
                "cloudflare",
                "ddos protection",
                "just a moment",
                "please wait"
            ]
            
            if any(keyword in page_content for keyword in verification_keywords):
                logger.info("Security verification detected, waiting for resolution...")
                
                # Wait for verification to complete
                if self._wait_for_captcha_resolution(max_wait_time=90):
                    logger.info("Verification completed, proceeding with scraping")
                    
                    # Additional wait for content to fully load after verification
                    time.sleep(5)
                    
                    # Try to wait for actual content to appear
                    try:
                        WebDriverWait(self.driver, 20).until(
                            lambda driver: len(driver.find_elements(By.TAG_NAME, "p")) > 0 or
                                        len(driver.find_elements(By.TAG_NAME, "article")) > 0 or
                                        len(driver.find_elements(By.CSS_SELECTOR, "[class*='content']")) > 0
                        )
                        logger.info("Content elements detected after verification")
                    except TimeoutException:
                        logger.warning("Content elements not detected, proceeding anyway")
                    
                else:
                    logger.warning("Verification did not complete within timeout, proceeding anyway")
            
            # Additional wait for any dynamic content to load
            time.sleep(random.uniform(3, 7))  # Random wait to appear more human-like
            
            # Try to detect if content is still loading
            self._wait_for_content_stability()
            
            # Get page source after all waiting
            html_source = self.driver.page_source
            raw_html_length = len(html_source)
            
            # Check if we still have verification content
            if any(keyword in html_source.lower() for keyword in verification_keywords):
                logger.warning("Still seeing verification content, scraping may be incomplete")
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_source, 'html.parser')
            
            # Extract title
            title = self.driver.title or urlparse(url).netloc
            
            # Debug: Count paragraphs before cleaning
            initial_p_count = len(soup.find_all('p'))
            logger.debug(f"Found {initial_p_count} paragraph tags in dynamic HTML")
            
            # Remove unwanted elements
            soup = self._remove_unwanted_elements(soup)
            
            # Debug: Count paragraphs after cleaning
            cleaned_p_count = len(soup.find_all('p'))
            logger.debug(f"Paragraphs remaining after cleaning: {cleaned_p_count}")
            
            # Extract content intelligently
            text_content, paragraph_count = self._extract_content_intelligently(soup)
            
            # Check if we actually got meaningful content
            if len(text_content.split()) < 50:
                logger.warning("Very little content extracted, verification may not have completed")
            
            word_count = len(text_content.split())
            char_count = len(text_content)
            
            logger.info(f"Dynamic scraping completed. Words: {word_count}, Characters: {char_count}, Paragraphs: {paragraph_count}")
            
            return ScrapedContent(
                url=url,
                title=title,
                content=text_content,
                word_count=word_count,
                char_count=char_count,
                scraped_at=datetime.now().isoformat(),
                scraping_method="dynamic_with_verification",
                success=True,
                raw_html_length=raw_html_length,
                paragraphs_found=paragraph_count
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
                scraping_method="dynamic_with_verification",
                success=False,
                error_message=str(e)
            )
    
    def scrape_url(self, url: str) -> ScrapedContent:
        """
        Scrape a URL with automatic fallback from static to dynamic scraping.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedContent object with scraped data
        """
        try:
            if self.config.force_dynamic:
                return self._scrape_dynamic_content(url)
            
            # Try static scraping first
            static_result = self._scrape_static_content(url)
            
            # If static scraping failed or returned minimal content, try dynamic
            if not static_result.success or static_result.word_count < 100:
                logger.info("Static scraping failed or returned minimal content, trying dynamic scraping")
                dynamic_result = self._scrape_dynamic_content(url)
                
                # Return the better result
                if dynamic_result.success and dynamic_result.word_count > static_result.word_count:
                    return dynamic_result
            
            return static_result
        
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
                error_message=str(e)
            )
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self._driver_initialized = False
        
        if self.session:
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Utility functions for backend integration

def scrape_single_url(url: str, config: ScrapingConfig = None) -> ScrapedContent:
    """
    Scrape a single URL with automatic cleanup.
    
    Args:
        url: URL to scrape
        config: Optional scraping configuration
        
    Returns:
        ScrapedContent object with scraped data
    """
    with WebScraper(config) as scraper:
        return scraper.scrape_url(url)

def scrape_multiple_urls(urls: List[str], config: ScrapingConfig = None) -> List[ScrapedContent]:
    """
    Scrape multiple URLs efficiently with shared resources.
    
    Args:
        urls: List of URLs to scrape
        config: Optional scraping configuration
        
    Returns:
        List of ScrapedContent objects
    """
    results = []
    
    with WebScraper(config) as scraper:
        for url in urls:
            try:
                result = scraper.scrape_url(url)
                results.append(result)
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

def scrape_to_dict(url: str, config: ScrapingConfig = None) -> Dict:
    """
    Scrape a URL and return result as dictionary.
    
    Args:
        url: URL to scrape
        config: Optional scraping configuration
        
    Returns:
        Dictionary containing scraped data
    """
    result = scrape_single_url(url, config)
    return asdict(result)

def scrape_to_json(url: str, config: ScrapingConfig = None) -> str:
    """
    Scrape a URL and return result as JSON string.
    
    Args:
        url: URL to scrape
        config: Optional scraping configuration
        
    Returns:
        JSON string containing scraped data
    """
    result = scrape_single_url(url, config)
    return json.dumps(asdict(result), indent=2, ensure_ascii=False)

# Example usage functions for different scenarios

def scrape_for_content_only(url: str, max_length: int = 50000) -> str:
    """
    Scrape a URL and return only the content text.
    
    Args:
        url: URL to scrape
        max_length: Maximum content length
        
    Returns:
        Extracted content text
    """
    config = ScrapingConfig(max_content_length=max_length)
    result = scrape_single_url(url, config)
    return result.content if result.success else ""

def scrape_with_metadata(url: str) -> Dict[str, Union[str, int, bool]]:
    """
    Scrape a URL and return content with metadata.
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary with content and metadata
    """
    result = scrape_single_url(url)
    return {
        'url': result.url,
        'title': result.title,
        'content': result.content,
        'word_count': result.word_count,
        'success': result.success,
        'scraped_at': result.scraped_at,
        'error_message': result.error_message
    }

def quick_scrape(url: str, timeout: int = 15) -> ScrapedContent:
    """
    Quick scrape with minimal waiting and faster timeout.
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        ScrapedContent object
    """
    config = ScrapingConfig(
        timeout=timeout,
        wait_for_stability=False,
        handle_verification=False
    )
    return scrape_single_url(url, config)

# Backend integration example
class WebScrapingService:
    """
    Service class for web scraping in backend applications.
    """
    
    def __init__(self, default_config: ScrapingConfig = None):
        self.default_config = default_config or ScrapingConfig()
        self.active_scrapers = {}
    
    def scrape(self, url: str, config: ScrapingConfig = None) -> ScrapedContent:
        """Main scraping method."""
        config = config or self.default_config
        return scrape_single_url(url, config)
    
    def batch_scrape(self, urls: List[str], config: ScrapingConfig = None) -> List[ScrapedContent]:
        """Batch scraping method."""
        config = config or self.default_config
        return scrape_multiple_urls(urls, config)
    
    def get_content_only(self, url: str) -> str:
        """Get only content text."""
        return scrape_for_content_only(url)
    
    def health_check(self) -> Dict[str, bool]:
        """Check if scraping service is healthy."""
        try:
            # Test with a simple static site
            test_url = "https://httpbin.org/html"
            result = quick_scrape(test_url)
            return {
                'healthy': result.success,
                'static_scraping': result.success,
                'dynamic_scraping': True  # Would need a more complex test
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'static_scraping': False,
                'dynamic_scraping': False,
                'error': str(e)
            }
    
    def cleanup(self):
        """Clean up any active scrapers."""
        for scraper in self.active_scrapers.values():
            try:
                scraper.close()
            except Exception as e:
                logger.error(f"Error cleaning up scraper: {e}")
        self.active_scrapers.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Main execution example
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html"
    ]
    
    # Create service
    service = WebScrapingService()
    
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
            print(f"Method: {result.scraping_method}")
            if not result.success:
                print(f"Error: {result.error_message}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    # Test batch scraping
    try:
        batch_results = service.batch_scrape(test_urls)
        print(f"\nBatch scraping completed: {len(batch_results)} results")
        for i, result in enumerate(batch_results):
            print(f"  {i+1}. {result.url} - Success: {result.success}")
    except Exception as e:
        print(f"Batch scraping error: {e}")
    
    # Clean up
    service.cleanup()
    print("\nScraping service cleaned up successfully")