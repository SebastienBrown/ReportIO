from scripts.llm.chat import chat_llm
from langchain.schema import SystemMessage, HumanMessage
import re
from typing import Dict


def generate_answer_from_context(query: str, top_urls:list[str], retrieved_chunks: list[dict[str,str]]) -> str:
    """Uses LLM to generate an answer based on the retrieved context"""
    

    # Build context with citation markers: add [URL] inline after each chunk
    context_blocks = [
        f"{chunk['text'].strip()} [{chunk['url'].strip()}]"
        for chunk in retrieved_chunks if chunk.get("text") and chunk.get("url")
    ]
    context = "\n\n".join(context_blocks)

    print("Context is ",context)

    ######################
    

    final_top_urls=list({chunk["url"] for chunk in retrieved_chunks if "url" in chunk})
    print("TOP URLS ARE ",final_top_urls,"\n\n\n")

    urls_string = ", ".join(final_top_urls)

    system_prompt =  f"""You are an expert research assistant and technical report writer. Your task is to write a long-form, structured chapter "
    "on a specific subtopic of a larger research question. You will be given a chapter title and a collection of context documents, "
    "each with a url identifier - you should cite those to support your claims.\n\n"

    "**Your chapter must:**\n\n"
    "You must style all headers as bold markdown"
    "1. **Introduction**\n"
    "   - Begin with a concise introduction (2–3 sentences) that previews what the section will cover.\n\n"
    "2. **Main Body (2–3 Subsections)**\n"
    "   - Include 2 to 3 subsections, each with a descriptive subheading, each formatted as bold markdown.\n"
    "   - Each subsection should:\n"
    "     • Use evidence, examples, or data from the provided sources.\n"
    "     •  Clearly cite the source(s) directly by URL — use the actual link in square brackets like [https://example.com/source1].\n"
+   " • If multiple sources support the same point, cite them together by their full URLs (e.g., [https://example.com/1][https://example.com/2]).\n"
    "     • Compare differing viewpoints if available, and highlight gaps where applicable.\n\n"
    "Only use a citation as a source if it in the following list {urls_string}"
    "3. **Summary**\n"
    "   - End the chapter with a short summary or takeaway paragraph (2–4 sentences).\n"
    "   - Reinforce the most important findings or reflections.\n\n"
    "**Additional Requirements:**\n"
    "- Do not add any information that isn’t grounded in the provided sources.\n"
    "- If the provided context lacks sufficient information to cover the chapter title, say:\n"
    "  'The provided context does not contain enough relevant information to write this section.'\n\n"
    "**Formatting:**\n"
    "- Use markdown-style formatting:\n"
    "  • Chapter title as H2 (`##`) and bold ('**X**')\n"
    "  • Subsection titles as H3 (`###`) and bold ('**X**')\n"
    "  • Separate paragraphs with double line breaks\n"
    "  • Bold the headers **example**, and add an additional line break between sections and sub sections"
    "  • Number the subsections as a,b,c as a markdown list"
    "- Keep tone factual, analytical, and accessible to a general expert audience.\n"

    "If the provided context lacks sufficient information to cover the chapter title, say:\n"
    "'The provided context does not contain enough relevant information to write this section.'\n"

    "Example citation style:"
    "Green hydrogen is produced via electrolysis using renewable power [https://energy.gov/green-hydrogen-overview]."
    "This is how all sources should be cited in the final output."""


    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    response = chat_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    print("system prompt is ",system_prompt)
    print("response is ",response)

    url_map = {}
    numbered_context = response.content
    next_index = 1

    for url in final_top_urls:
        if url not in url_map:
            url_map[url] = next_index
            next_index += 1

        # Replace [URL] with [number]
        numbered_context = numbered_context.replace(f'[{url}]', f'[{url_map[url]}]')
    
    print("NUMBERED CONTEXT IS ",numbered_context)
    
    # Invert the url_map to get number -> url mapping
    number_to_url = {v: k for k, v in url_map.items()}

    def linkify_citations(text: str, number_to_url: dict[int, str]) -> str:
        """
        Replaces [1], [2], etc. with markdown hyperlinks like [1](https://example.com)
        """
        def replace_number(match):
            num = int(match.group(1))
            if num in number_to_url:
                return f"[\\[{num}\\]]({number_to_url[num]})"
            return match.group(0)

        return re.sub(r'\[(\d+)\]', replace_number, text)


    linked_text = linkify_citations(numbered_context, number_to_url)
    print("LINKED TEXT IS ",linked_text)

    #numbered_text = replace_citation_numbers_with_links(numbered_context, url_map)

   # print("numbered text is ",numbered_text,"\n\n\n")

    ######################
    
    return linked_text

def generate_gpt_answer(query: str) -> str:
    """Generates a direct GPT response without RAG context."""
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer clearly and concisely."),
        HumanMessage(content=query)
    ]
    response = chat_llm.invoke(messages)
    return response.content

def replace_citation_numbers_with_links(text: str, url_map: Dict[int, str]) -> str:
    """
    Replaces [1], [2], etc. in the text with markdown hyperlinks like [1](https://...).
    
    Args:
        text: Text containing numbered citations like [1], [2].
        url_map: Dict mapping citation numbers to their URLs.
    
    Returns:
        Text with markdown-style citation links.
    """
    def replace_match(match):
        num = int(match.group(1))
        if num in url_map:
            return f"[{num}]({url_map[num]})"
        return match.group(0)  # leave as-is if not found

    return re.sub(r'\[(\d+)\]', replace_match, text)



__all__ = ["chat_llm", "generate_gpt_answer"]
