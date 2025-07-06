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
import argparse
import os
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

class WebScraper:
    """Universal web scraper that handles both static and dynamic content."""
    
    def __init__(self, 
                 timeout: int = 30,
                 headless: bool = True,
                 user_agent: str = None):
        """
        Initialize the web scraper.
        
        Args:
            timeout: Request timeout in seconds
            headless: Whether to run browser in headless mode
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.headless = headless
        self.user_agent = user_agent or (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        
        # Session for static scraping
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
        # Selenium driver (initialized when needed)
        self.driver = None
    
    def _setup_selenium_driver(self):
        """Enhanced Selenium setup with better anti-detection."""
        if self.driver is not None:
            return
            
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            
            # Enhanced anti-detection options
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={self.user_agent}')
            
            # Additional anti-detection measures
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Enhanced Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            logger.error("Make sure ChromeDriver is installed and in your PATH")
            raise
    
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
                logger.info(f"Found main content using selector: {selector}")
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
                logger.info(f"Found main content with score {max_content_score:.1f}")
        
        # Strategy 3: Fall back to body if nothing else works
        if not main_content:
            main_content = soup.find('body')
            logger.info("Using body as main content")
        
        if not main_content:
            main_content = soup
            logger.info("Using entire soup as main content")
        
        # Extract text content comprehensively
        text_parts = []
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
            logger.info("Using comprehensive text extraction method")
            
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
            logger.info("Using fallback full text extraction")
            extracted_text = main_content.get_text(separator=' ', strip=True)
            extracted_text = self._cleanup_text(extracted_text)
        
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
        logger.info("Waiting for content to stabilize...")
        
        previous_content_hash = None
        stable_count = 0
        
        for i in range(max_wait // 2):  # Check every 2 seconds
            try:
                current_content = self.driver.page_source
                current_hash = hash(current_content)
                
                if current_hash == previous_content_hash:
                    stable_count += 1
                    if stable_count >= 3:  # Content unchanged for 6 seconds
                        logger.info("Content appears stable")
                        return True
                else:
                    stable_count = 0
                    previous_content_hash = current_hash
                
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error checking content stability: {e}")
                time.sleep(2)
        
        logger.info("Content stability check timeout")
        return False
    
    def scrape_static_content(self, url: str) -> ScrapedContent:
        """
        Scrape static HTML content using requests and BeautifulSoup.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedContent object with scraped data
        """
        try:
            logger.info(f"Attempting static scraping for: {url}")
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            raw_html_length = len(response.content)
            logger.info(f"Retrieved HTML content: {raw_html_length} bytes")
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else urlparse(url).netloc
            
            # Debug: Count paragraphs before cleaning
            initial_p_count = len(soup.find_all('p'))
            logger.info(f"Found {initial_p_count} paragraph tags in raw HTML")
            
            # Remove unwanted elements
            soup = self._remove_unwanted_elements(soup)
            
            # Debug: Count paragraphs after cleaning
            cleaned_p_count = len(soup.find_all('p'))
            logger.info(f"Paragraphs remaining after cleaning: {cleaned_p_count}")
            
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
    
    def wait_for_captcha_resolution(self, max_wait_time: int = 60) -> bool:
        """
        Wait for CAPTCHA or security verification to complete.
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            bool: True if verification appears to be complete, False if timeout
        """
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
                    logger.info("Page content changed, verification may be progressing...")
                    last_content = current_content
                
                # Wait a bit before checking again
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error while waiting for verification: {e}")
                time.sleep(2)
        
        logger.warning(f"Verification wait timeout after {max_wait_time} seconds")
        return False

    def scrape_dynamic_content(self, url: str) -> ScrapedContent:
        """
        Enhanced dynamic scraping with CAPTCHA handling.
        """
        try:
            logger.info(f"Attempting dynamic scraping for: {url}")
            
            # Initialize Selenium driver if not already done
            if self.driver is None:
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
                if self.wait_for_captcha_resolution(max_wait_time=90):
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
            logger.info(f"Found {initial_p_count} paragraph tags in dynamic HTML")
            
            # Remove unwanted elements
            soup = self._remove_unwanted_elements(soup)
            
            # Debug: Count paragraphs after cleaning
            cleaned_p_count = len(soup.find_all('p'))
            logger.info(f"Paragraphs remaining after cleaning: {cleaned_p_count}")
            
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
    
    def scrape_url(self, url: str, force_dynamic: bool = False) -> ScrapedContent:
        """
        Scrape a URL with automatic fallback from static to dynamic scraping.
        
        Args:
            url: URL to scrape
            force_dynamic: Force use of dynamic scraping (Selenium)
            
        Returns:
            ScrapedContent object with scraped data
        """
        if force_dynamic:
            return self.scrape_dynamic_content(url)
        
        # Try static scraping first
        static_result = self.scrape_static_content(url)
        
        # If static scraping failed or returned minimal content, try dynamic
        if not static_result.success or static_result.word_count < 100:
            logger.info("Static scraping failed or returned minimal content, trying dynamic scraping")
            dynamic_result = self.scrape_dynamic_content(url)
            
            # Return the better result
            if dynamic_result.success and dynamic_result.word_count > static_result.word_count:
                return dynamic_result
        
        return static_result
    
    def save_to_json(self, content: ScrapedContent, output_file: str):
        """
        Save scraped content to JSON file.
        
        Args:
            content: ScrapedContent object to save
            output_file: Path to output JSON file
        """
        try:
            # Convert dataclass to dictionary
            content_dict = asdict(content)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Write to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(content_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Content saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            raise
    
    def analyze_content_types(self, url: str):
        """
        Analyze what types of content elements exist on the page.
        
        Args:
            url: URL to analyze
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print(f"\n=== Content Type Analysis for {url} ===")
            
            # Analyze different content types
            content_types = {
                'paragraphs': soup.find_all('p'),
                'headings': soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']),
                'lists': soup.find_all(['ul', 'ol']),
                'list_items': soup.find_all('li'),
                'blockquotes': soup.find_all('blockquote'),
                'tables': soup.find_all('table'),
                'table_cells': soup.find_all(['td', 'th']),
                'spans': soup.find_all('span'),
                'divs_with_text': [div for div in soup.find_all('div') if div.get_text(strip=True) and not div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
                'strong_emphasis': soup.find_all(['strong', 'b', 'em', 'i']),
            }
            
            print("Content element counts:")
            for content_type, elements in content_types.items():
                print(f"  {content_type}: {len(elements)}")
            
            # Show sample content from each type
            print("\nSample content from each type:")
            for content_type, elements in content_types.items():
                if elements:
                    sample = elements[0].get_text(strip=True)
                    print(f"  {content_type}: {sample[:100]}...")
            
            # Analyze text distribution
            body = soup.find('body')
            if body:
                total_text = body.get_text(strip=True)
                para_text = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
                non_para_text_ratio = (len(total_text) - len(para_text)) / len(total_text) if total_text else 0
                
                print(f"\nText distribution:")
                print(f"  Total text length: {len(total_text)}")
                print(f"  Paragraph text length: {len(para_text)}")
                print(f"  Non-paragraph text ratio: {non_para_text_ratio:.2%}")
                
                if non_para_text_ratio > 0.3:
                    print("  ⚠️  Significant non-paragraph text detected!")
                    
        except Exception as e:
            print(f"Error in content type analysis: {e}")

    def debug_html_structure(self, url: str):
        """
        Debug function to analyze HTML structure and identify content extraction issues.
        
        Args:
            url: URL to analyze
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print(f"\n=== HTML Structure Analysis for {url} ===")
            print(f"Total HTML size: {len(response.content)} bytes")
            print(f"Total paragraphs: {len(soup.find_all('p'))}")
            print(f"Total divs: {len(soup.find_all('div'))}")
            print(f"Total links: {len(soup.find_all('a'))}")
            
            # Analyze paragraph distribution
            containers = soup.find_all(['div', 'section', 'article', 'main'])
            print(f"\nParagraph distribution:")
            for i, container in enumerate(containers[:10]):  # Show top 10
                p_count = len(container.find_all('p'))
                if p_count > 0:
                    classes = container.get('class', [])
                    id_attr = container.get('id', '')
                    print(f"  Container {i}: {p_count} paragraphs, classes: {classes}, id: {id_attr}")
            
            # Show sample paragraphs
            paragraphs = soup.find_all('p')
            print(f"\nSample paragraphs (first 3):")
            for i, p in enumerate(paragraphs[:3]):
                text = p.get_text(strip=True)
                print(f"  P{i}: {text[:100]}...")
                
            # Run content type analysis
            self.analyze_content_types(url)
            
        except Exception as e:
            print(f"Error in debug analysis: {e}")
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None
        
        if self.session:
            self.session.close()

def main():
    """Main function to run the web scraper from command line."""
    parser = argparse.ArgumentParser(description='Universal Web Scraper')
    parser.add_argument('url', help='URL to scrape')
    parser.add_argument('-o', '--output', default='scraped_content.json', 
                        help='Output JSON file (default: scraped_content.json)')
    parser.add_argument('-t', '--timeout', type=int, default=30,
                        help='Request timeout in seconds (default: 30)')
    parser.add_argument('--force-dynamic', action='store_true',
                        help='Force use of dynamic scraping (Selenium)')
    parser.add_argument('--no-headless', action='store_true',
                        help='Run browser in non-headless mode')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with HTML structure analysis')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze content types without scraping')
    parser.add_argument('--user-agent', type=str,
                        help='Custom user agent string')
    
    args = parser.parse_args()
    
    # Create scraper instance
    scraper = WebScraper(
        timeout=args.timeout,
        headless=not args.no_headless,
        user_agent=args.user_agent
    )
    
    try:
        if args.analyze:
            # Just analyze content types
            scraper.analyze_content_types(args.url)
        elif args.debug:
            # Debug HTML structure
            scraper.debug_html_structure(args.url)
        else:
            # Scrape the content
            print(f"Scraping: {args.url}")
            result = scraper.scrape_url(args.url, force_dynamic=args.force_dynamic)
            
            # Print results
            if result.success:
                print(f"\n✅ Scraping successful!")
                print(f"Title: {result.title}")
                print(f"Word count: {result.word_count}")
                print(f"Character count: {result.char_count}")
                print(f"Paragraphs found: {result.paragraphs_found}")
                print(f"Scraping method: {result.scraping_method}")
                print(f"Raw HTML length: {result.raw_html_length}")
                
                # Save to file
                scraper.save_to_json(result, args.output)
                
                # Show content preview
                preview = result.content[:500] + "..." if len(result.content) > 500 else result.content
                print(f"\nContent preview:\n{preview}")
                
            else:
                print(f"\n❌ Scraping failed: {result.error_message}")
                
    except KeyboardInterrupt:
        print("\n⚠️  Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Clean up
        scraper.close()

if __name__ == "__main__":
    main()