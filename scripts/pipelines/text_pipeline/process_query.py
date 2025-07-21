import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from scripts.llm.chat import chat_llm
import re

# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

class QueryTransformer:
    def __init__(self):
        # Question words to remove
        self.question_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose',
            'analyze', 'evaluate', 'examine', 'assess', 'discuss', 'explain',
            'describe', 'consider', 'review', 'explore'
        }
        
        # Articles and common prepositions to remove
        self.articles_prepositions = {
            'the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
            'from', 'to', 'into', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
        
        # Verbs to remove (common ones that don't add search value)
        self.verbs_to_remove = {
            'are', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'must', 'shall', 'play', 'plays', 'played',
            'impact', 'impacts', 'impacted', 'affect', 'affects', 'affected',
            'influence', 'influences', 'influenced', 'shape', 'shapes', 'shaped',
            'reshape', 'reshapes', 'reshaped', 'reshaping'
        }
        
        # Words to keep even if they might normally be filtered
        self.keep_words = {
            'post', 'pre', 'anti', 'pro', 'non', 'multi', 'inter', 'trans',
            'cyber', 'ai', 'strategic', 'military', 'nuclear', 'global',
            'modern', 'enabled', 'authoritarian', 'private', 'middle', 'east'
        }
        
        # Compound terms to preserve as units
        self.compound_terms = [
            'middle east', 'east asia', 'south china sea', 'taiwan strait',
            'nuclear proliferation', 'cyber warfare', 'cyberwarfare',
            'military doctrine', 'civil liberties', 'private military',
            'abraham accords', 'strategic rivalry', 'global trade',
            'rare earths', 'ai-enabled', 'surveillance impact'
        ]

    def preserve_compounds(self, text):
        """Replace compound terms with placeholder tokens to preserve them"""
        text_lower = text.lower()
        preserved = text_lower
        replacements = {}
        
        for i, compound in enumerate(self.compound_terms):
            if compound in text_lower:
                placeholder = f"__COMPOUND_{i}__"
                preserved = preserved.replace(compound, placeholder)
                replacements[placeholder] = compound
        
        return preserved, replacements

    def restore_compounds(self, text, replacements):
        """Restore compound terms from placeholders"""
        for placeholder, compound in replacements.items():
            text = text.replace(placeholder, compound)
        return text

    def is_proper_noun(self, word, pos_tag):
        """Check if word is a proper noun"""
        return pos_tag in ['NNP', 'NNPS'] or word[0].isupper()

    def should_keep_word(self, word, pos_tag):
        """Determine if a word should be kept based on various criteria"""
        word_lower = word.lower()
        
        # Always keep if in keep_words list
        if word_lower in self.keep_words:
            return True
            
        # Always keep proper nouns
        if self.is_proper_noun(word, pos_tag):
            return True
            
        # Keep important nouns
        if pos_tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            return True
            
        # Keep important adjectives
        if pos_tag in ['JJ', 'JJR', 'JJS'] and word_lower not in self.articles_prepositions:
            return True
            
        # Remove question words
        if word_lower in self.question_words:
            return False
            
        # Remove articles and prepositions
        if word_lower in self.articles_prepositions:
            return False
            
        # Remove common verbs
        if word_lower in self.verbs_to_remove:
            return False
            
        # Keep numbers and special characters
        if word.isdigit() or not word.isalpha():
            return True
            
        return False

    def clean_punctuation(self, text):
        """Clean up punctuation while preserving hyphens in compound words"""
        # Remove question marks and other end punctuation
        text = re.sub(r'[?!.]+$', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def transform_query(self, query):
        """Transform a natural language query into search keywords"""
        # Clean initial punctuation
        query = self.clean_punctuation(query)
        
        # Preserve compound terms
        preserved_query, replacements = self.preserve_compounds(query)
        
        # Tokenize and tag
        tokens = word_tokenize(preserved_query)
        pos_tags = pos_tag(tokens)
        
        # Filter tokens based on our rules
        filtered_tokens = []
        for token, pos in pos_tags:
            if self.should_keep_word(token, pos):
                filtered_tokens.append(token)
        
        # Join tokens back into a string
        result = ' '.join(filtered_tokens)
        
        # Restore compound terms
        result = self.restore_compounds(result, replacements)
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result

    def transform_multiple(self, queries):
        """Transform multiple queries"""
        return [self.transform_query(query) for query in queries]

def parse_list(text: str) -> list[str]:
    """
    Parses a numbered list from a string into a Python list of strings.
    Supports formats like:
    1. Item
    2) Item
    """
    lines = text.strip().split("\n")
    result = []
    for line in lines:
        match = re.match(r"^\s*\d+[\.\)]\s*(.*)", line)
        if match:
            result.append(match.group(1).strip())
    return result

def generate_sections(prompt: str) -> list[str]:
    SYSTEM = "You are an expert content planner."
    USER = f"""
    Break down the following topic into 4 to 6 logically ordered sections.
    Topic: "{prompt}"
    Return only the section titles as a numbered list.
    """
    response = chat_llm.invoke([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER}
    ])
    return parse_list(response.content)

# Example usage and testing
if __name__ == "__main__":
    transformer = QueryTransformer()
    
    # Test queries from the original example
    test_queries = [
        "What were the most significant outcomes of the Congress of Vienna?",
        "How did the fall of the Ottoman Empire reshape the Middle East?",
        "Compare the French and American Revolutions in political outcomes.",
        "What are the historical roots of authoritarianism in Eastern Europe?",
        "Analyze how Roman imperial logistics contributed to longevity.",
        "What are the long-term effects of colonial education systems in Africa?",
        "How did the Cold War shape the modern international system?",
        "Compare Leninist vs Maoist applications of communism.",
        "How has populism evolved in the 21st century?",
        "What political models emerged from the Enlightenment?"
    ]
    
    print("Original Query → Transformed Query")
    print("=" * 60)
    
    for query in test_queries:
        transformed = transformer.transform_query(query)
        print(f"{query}")
        print(f"→ {transformed}")
        print()

    # Test with custom queries
    print("\nCustom Test Queries:")
    print("=" * 30)
    
    custom_queries = [
        "How does climate change affect international migration patterns?",
        "What is the impact of social media on political polarization?",
        "Why are semiconductor supply chains so vulnerable to disruption?"
    ]
    
    for query in custom_queries:
        transformed = transformer.transform_query(query)
        print(f"{query}")
        print(f"→ {transformed}")
        print()