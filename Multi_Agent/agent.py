from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from operator import add as add_messages
from typing import TypedDict, Optional,Annotated, Sequence, List, Dict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from firecrawl import Firecrawl
from firecrawl.v2.types import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import os   
import httpx  
import traceback 
from typing import Optional

class OnPageResult(TypedDict):
    """A data model for storing the results of an on-page SEO analysis."""
    url: str
    title: Optional[str] 
    meta_description: Optional[str] 
    headings: Dict[str, List[str]] 
    body_text_length: int = 0
    images: List[Dict[str, Optional[str]]] 
    links: Dict[str, List[str]] 
    error_message: Optional[str] 

class CoreWebVitals(TypedDict):
    """Data model for Core Web Vitals metrics."""
    lcp: Optional[float] # Largest Contentful Paint (in seconds)
    fid: Optional[float] # First Input Delay (in milliseconds)
    cls: Optional[float] # Cumulative Layout Shift (unitless score)

class TechnicalAuditResult(TypedDict):
    """Final complete audit report."""
    url:str
    performance_score: Optional[int] = None
    mobile_friendly: Optional[bool] = None
    uses_https: Optional[bool] = None
    core_web_vitals: CoreWebVitals 
    error_message: Optional[str] = None

load_dotenv()

class CrawlState(TypedDict):
    url: str
    status_code: int
    html_content: Optional[str]
    extracted_text: Optional[str]
    error_message: Optional[str]



class AgentState(TypedDict,CrawlState):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    crawl_result : CrawlState
    technicalAuditResult: TechnicalAuditResult


class OnPageAnalyzer:
    """
    Parses HTML content to extract key on-page SEO elements.
    """
    def __init__(self, url: str, html_content: str):
        self.url = url
        self.base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        self.soup = BeautifulSoup(html_content, 'lxml')

    def analyze(self) -> OnPageResult:
        """Runs all extraction methods and returns a structured result."""
        try:
            headings = self._extract_headings()
            images = self._extract_images()
            links = self._extract_links()

            body_text = (
                self.soup.body.get_text(separator=' ', strip=True)
                if self.soup.body else ""
            )

            return OnPageResult(
                url=self.url,
                title=self._extract_title(),
                meta_description=self._extract_meta_description(),
                headings=headings,
                body_text_length=len(body_text.split()),
                images=images,
                links=links
            )
        except Exception as e:
            return OnPageResult(
                url=self.url,
                error_message=f"Parsing error: {str(e)}"
            )

    def _extract_title(self) -> Optional[str]:
        return self.soup.title.string.strip() if self.soup.title else None

    def _extract_meta_description(self) -> Optional[str]:
        meta_tag = self.soup.find('meta', attrs={'name': 'description'})
        return meta_tag.get('content').strip() if meta_tag else None

    def _extract_headings(self) -> Dict[str, List[str]]:
        headings = {}
        for i in range(1, 7):
            tag = f'h{i}'
            found = [h.get_text(strip=True) for h in self.soup.find_all(tag)]
            if found:
                headings[tag] = found
        return headings

    def _extract_images(self) -> List[Dict[str, Optional[str]]]:
        images = []
        for img in self.soup.find_all('img'):
            images.append({
                'src': img.get('src'),
                'alt': img.get('alt', '').strip() or None
            })
        return images

    def _extract_links(self) -> Dict[str, List[str]]:
        links = {"internal": [], "external": []}
        for a in self.soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(self.base_url, href)
            if self.base_url in absolute_url:
                links["internal"].append(absolute_url)
            else:
                links["external"].append(absolute_url)
        return links


@tool
def crawl_webpage_tool(url: str) -> CrawlState:
    """Crawl a webpage using Firecrawl and extract HTML + Markdown content."""

    api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        raise ValueError(
            "Error: FIRECRAWL_API_KEY is missing.\n"
            "Please set it in the environment variables or .env file."
        )

    client = Firecrawl(api_key=api_key)

    try:
        response = client.scrape(
            url,
            formats=["html", "markdown"]
        )

        # New Firecrawl Document class
        if isinstance(response, Document):
            return CrawlState(
                url=url,
                status_code=200,
                html_content=response.html,
                extracted_text=response.markdown,
                error_message=None
            )

        # Legacy dict response
        elif isinstance(response, dict):
            return CrawlState(
                url=url,
                status_code=200,
                html_content=response.get("html"),
                extracted_text=response.get("markdown"),
                error_message=None
            )

        else:
            return CrawlState(
                url=url,
                status_code=500,
                html_content=None,
                extracted_text=None,
                error_message=f"Unknown response type: {type(response)}"
            )

    except Exception as e:
        return CrawlState(
            url=url,
            status_code=500,
            html_content=None,
            extracted_text=None,
            error_message=str(e)
        )
    

@tool("technical_seo_audit")
def technical_seo_audit(url: str) -> dict:
    """
    Runs a full technical SEO audit using the Google PageSpeed Insights API.
    """

    api_key = os.getenv("PAGESPEED_API_KEY")
    if not api_key:
        return {
            "url": url,
            "error_message": "Missing PAGESPEED_API_KEY environment variable."
        }

    base_url = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

    params = {
        "url": url,
        "key": api_key,
        "strategy": "MOBILE"
    }

    # --- FIX: Use httpx.Client (Synchronous) instead of AsyncClient ---
    # We also use a 'with' block to ensure the connection closes properly.
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status() # Now works because response is a real Response object
            data = response.json()

    except httpx.ReadTimeout:
        return TechnicalAuditResult(
            url=url,
            error_message="Request timed out while fetching PageSpeed Insights data."
        ).model_dump() # Updated to .model_dump() for Pydantic V2 compatibility

    except httpx.HTTPStatusError as e:
        return TechnicalAuditResult(
            url=url,
            error_message=f"HTTP error {e.response.status_code}: {e.response.text}"
        ).model_dump()

    except Exception as e:
        traceback.print_exc()
        return TechnicalAuditResult(
            url=url,
            error_message=f"Unexpected error: {str(e)}"
        ).model_dump()

    lighthouse = data.get("lighthouseResult", {})
    categories = lighthouse.get("categories", {})
    audits = lighthouse.get("audits", {})

    performance_score = int(categories.get("performance", {}).get("score", 0) * 100)
    uses_https = audits.get("is-on-https", {}).get("score") == 1
    mobile_friendly = audits.get("mobile-friendly", {}).get("score") == 1

    lcp = audits.get("largest-contentful-paint", {}).get("numericValue", 0) / 1000
    fid = audits.get("max-potential-fid", {}).get("numericValue", 0)
    cls = audits.get("cumulative-layout-shift", {}).get("numericValue", 0)

    result = TechnicalAuditResult(
        url=url,
        performance_score=performance_score,
        uses_https=uses_https,
        mobile_friendly=mobile_friendly,
        core_web_vitals=CoreWebVitals(
            lcp=round(lcp, 2), 
            fid=int(fid), 
            cls=round(cls, 3)
        )
    )

    return result

@tool("on_page_seo_audit", return_direct=False)
def on_page_seo_audit(url: str, html_content: str) -> dict:
    """
    Performs a full On-Page SEO audit of a webpage's HTML.

    Args:
        url (str): The page URL.
        html_content (str): The raw HTML content.

    Returns:
        dict: A complete on-page SEO analysis (headings, links, images, metadata).
    """
    try:
        analyzer = OnPageAnalyzer(url, html_content)
        result = analyzer.analyze()
        return result.dict()

    except Exception as e:
        return {
            "url": url,
            "error_message": f"Tool error: {str(e)}"
        }

tools = [crawl_webpage_tool, technical_seo_audit, on_page_seo_audit]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    """Calls the LLM with the current messages and returns the updated state."""

    system_prompt = SystemMessage(
        content=f"""
            You are a Web Crawling + SEO Audit AI assistant. Your job is to help the user scrape webpages and
            extract information from URLs they provide.

            Your behavior rules:
            1. If the user asks to crawl/scrape → ALWAYS call `crawl_webpage_tool`.
            2. If the user asks for SEO audit, technical issues, page speed, meta tags, sitemap, robots.txt → ALWAYS call `technical_seo_audit`.
            3. If the user asks questions about the crawled content → answer using the CrawlState.
            4. If no crawl has happened yet and the user asks about webpage content → ask them for a URL.
            5 NEVER fabricate crawl output.
            """
    )

    # First interaction
    if not state["messages"]:
        user_input = "Hi! I can crawl any webpage for you. Please provide a URL to begin."
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to crawl or extract next? ")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content=user_input)

    # Combine messages
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    # Call LLM
    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")

    # Tool call debug
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {
        "messages": list(state["messages"]) + [user_message, response]
    }

def should_continue(state: AgentState) -> str:
    
    messages = state["messages"]
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if(isinstance(message,ToolMessage) and 
            "crawl_webpage" in message.content.lower()):
            return "end"
    
    return "continue"

def print_messages(messages):
    if not messages: return 
    for message in messages[:-3]:
        if(isinstance(message, ToolMessage)):
            print(f"\n TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n======== DRAFTER =========")

    state = {"messages":[]}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED ===== ")

if __name__ == "__main__":
    run_document_agent()