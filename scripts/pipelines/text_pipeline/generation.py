from scripts.llm.chat import chat_llm
from langchain.schema import SystemMessage, HumanMessage
import re
from typing import Dict
from urllib.parse import urlparse


def generate_answer_from_context(query: str, top_urls:list[str], retrieved_chunks: list[dict[str,str]]) -> str:
    """Uses LLM to generate an answer based on the retrieved context"""
    
    """     # Print each URL
    for chunk in retrieved_chunks:
        if chunk.get("url"):
            print("URL:", chunk["url"].strip())
            print("\n")

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

    urls_string = ", ".join(final_top_urls) """

     # -----------------------------
    # STEP 1 — Generate stable citation IDs per unique URL
    # -----------------------------
    citation_map = {}  # url -> number
    indexed_chunks = []
    next_id = 1

    for chunk in retrieved_chunks:
        url = chunk.get("url")
        text = chunk.get("text")
        print("TEXT FOR CHUNK IS :",text)
        print("url is ",url)

        if not url or not text:
            continue

        url = url.strip()
        text = text.strip()

        # Assign existing or new citation number
        if url not in citation_map:
            citation_map[url] = next_id
            next_id += 1

        citation_id = citation_map[url]
        indexed_chunks.append(f"{text} [{citation_id}]")

    # Build model input context
    context = "\n\n".join(indexed_chunks)

    # Reverse mapping: number -> URL
    number_to_url = {v: k for k, v in citation_map.items()}
    print("CITATION MAP IS ",citation_map)
    print("NUMBER TO URL ",number_to_url)

    system_prompt =  f"""You are an expert research assistant and technical report writer. Your task is to write a long-form, structured chapter "
    "on a specific subtopic of a larger research question. 
    You will be given:
    - A collection of numbered context excerpts you should cite to support your claims: [1], [2], [3] ...\n\n"

    "**Your chapter must:**\n\n"
    "You must style all headers as bold markdown"
    "1. **Introduction**\n"
    "   - Begin with a concise introduction (2–3 sentences) that previews what the section will cover.\n\n"
    "2. **Main Body (2–3 Subsections)**\n"
    "   - Include 2 to 3 subsections, each with a descriptive subheading, each formatted as bold markdown.\n"
    "   - Each subsection should:\n"
    "     • Use evidence, examples, or data from the provided sources.\n"
            RULES:
            - You may ONLY use citation numbers that appear in the source context.
            - Use citations ONLY by number: [1], [2], [3]
            - Never invent new citation numbers
            - Never rewrite or modify numbers
            - Never add URLs
            - If information needed is missing, say so directly.
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
    "Green hydrogen is produced via electrolysis using renewable power [1]."
    "This is how all sources should be cited in the final output.
    
    Emsure all markdown formatting is correct before returning."""


    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    response = chat_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    print("system prompt is ",system_prompt)
    print("response is ",response)

    raw_output = response.content
    raw_output = re.sub(r'\[(\d+)\]', lambda m: "" if int(m.group(1)) not in number_to_url else m.group(0), raw_output)


    # -----------------------------
    # STEP 5 — Replace [n] with [[n]](url)
    # -----------------------------

    print("CITATION MAP IS ",citation_map)
    print("NUMBER TO URL ",number_to_url)

    def linkify(text: str) -> str:
        def repl(match):
            num = int(match.group(1))
            if num in number_to_url:
                return f"[[{num}]]({number_to_url[num]})"
            return match.group(0)

        return re.sub(r"\[(\d+)\]", repl, text)

    final_output = linkify(raw_output)

    return final_output

    # url_map = {}
    # numbered_context = response.content
    # next_index = 1

    # def canonical_url(url: str) -> str:
    #     """Strips anchors, query params, trailing slashes."""
    #     parsed = urlparse(url)
    #     return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")

    # # Build canonical URL map
    # canonical_final_urls = {canonical_url(u): u for u in final_top_urls}

    # for url in final_top_urls:
    #     canon = canonical_url(url)
    #     if canon not in url_map:
    #         url_map[canon] = next_index
    #         next_index += 1

    #     # Replace all occurrences in the text, using canonical URL as key
    #     # Any URL in numbered_context will be canonicalized first for matching
    #     numbered_context = re.sub(
    #         re.escape(url) + r'([#?][^\]\s]*)?',  # match anchors or query params
    #         f'[{url_map[canon]}]',
    #         numbered_context
    #     )

    # print("NUMBERED CONTEXT IS ", numbered_context)
    
    # # Invert url_map to number -> original URL mapping
    # number_to_url = {v: canonical_final_urls[k] for k, v in url_map.items()}

    # # Step B: convert numbers to markdown links safely
    # def linkify_citations(text: str, number_to_url: dict[int, str]) -> str:
    #     """
    #     Replaces [1], [2], etc. with markdown hyperlinks like [1](https://example.com)
    #     """
    #     def replace_number(match):
    #         num = int(match.group(1))
    #         if num in number_to_url:
    #             return f"[{num}]({number_to_url[num]})"
    #         return match.group(0)

    #     return re.sub(r'\[(\d+)\]', replace_number, text)

    # linked_text = linkify_citations(numbered_context, number_to_url)
    # print("LINKED TEXT IS ", linked_text)

    # return linked_text


def generate_gpt_answer(query: str) -> str:
    """Generates a direct GPT response without RAG context."""
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer clearly and concisely."),
        HumanMessage(content=query)
    ]
    response = chat_llm.invoke(messages)
    return response.content

# def replace_citation_numbers_with_links(text: str, url_map: Dict[int, str]) -> str:
#     """
#     Replaces [1], [2], etc. in the text with markdown hyperlinks like [1](https://...).
    
#     Args:
#         text: Text containing numbered citations like [1], [2].
#         url_map: Dict mapping citation numbers to their URLs.
    
#     Returns:
#         Text with markdown-style citation links.
#     """
#     def replace_match(match):
#         num = int(match.group(1))
#         if num in url_map:
#             return f"[{num}]({url_map[num]})"
#         return match.group(0)  # leave as-is if not found

#     return re.sub(r'\[(\d+)\]', replace_match, text)



__all__ = ["chat_llm", "generate_gpt_answer"]
