from langgraph.graph import StateGraph,START ,END
from operator import add as add_messages
from typing import Any, TypedDict, Optional,Annotated, Sequence, List, Dict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from firecrawl import Firecrawl
from firecrawl.v2.types import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import os   
import requests
import traceback 
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from datetime import datetime

load_dotenv()

# Initialize the Gemini Model
# We use gemini-1.5-pro because it has a massive context window, 
# perfect for analyzing full page HTML/Markdown content.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2, # Low temperature for factual, analytical reporting
    convert_system_message_to_human=True # specific handling for Gemini API quirks
)

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages

# --- Pydantic Models (For LLM Structured Output) ---

class OptimizedTitleMeta(BaseModel):
    """Data model for a rewritten Title and Meta Description."""
    new_title: str = Field(description="The newly optimized, SEO-friendly title.")
    new_meta_description: str = Field(description="The newly optimized, compelling meta description.")

class GeneratedContentSection(BaseModel):
    """Data model for a newly generated content section."""
    suggested_heading: str = Field(description="A suitable H2 or H3 heading.")
    new_content_paragraph: str = Field(description="The content paragraph (100-150 words).")

# --- TypedDict (For LangGraph State) ---

class OptimizationResultState(TypedDict):
    """The dictionary stored in the main AgentState."""
    optimized_title_meta: Optional[Dict[str, str]] # Stored as dict, not Pydantic
    new_sections: List[Dict[str, str]] # List of dicts
    error_message: Optional[str]

class Recommendation(BaseModel):
    priority: str = Field(description="Priority: 'High', 'Medium', or 'Low'")
    category: str = Field(description="Category: 'Content', 'Technical', 'On-Page', 'Backlinks'")
    recommendation: str = Field(description="Specific action item.")
    justification: str = Field(description="Why this helps SEO.")

class SeoStrategy(BaseModel):
    recommendations: List[Recommendation]

class StrategyResult(TypedDict):
    recommendations: List[Dict[str, str]] # We store as list of dicts, not Pydantic objects
    error_message: Optional[str]

class KeywordReport(TypedDict):
    primary_keyword: str
    secondary_keywords: List[str]

class ContentGapReport(TypedDict):
    competitor_themes: List[str]
    suggested_topics: List[str]

class MarketResearchReport(TypedDict):
    """The final dictionary structure for the state."""
    keyword_analysis: Optional[KeywordReport]
    competitor_urls: List[str]
    content_gap_analysis: Optional[ContentGapReport]
    error_message: Optional[str]

class InternalKeywordSchema(BaseModel):
    primary_keyword: str = Field(description="The single most important keyword.")
    secondary_keywords: List[str] = Field(description="3-5 related keywords.")

class InternalGapSchema(BaseModel):
    competitor_themes: List[str] = Field(description="Common themes in competitor content.")
    suggested_topics: List[str] = Field(description="Topics missing from our page.")

class OnPageResult(TypedDict):
    """
    A data model for storing the results of an on-page SEO analysis.
    Updated to include Advanced SEO metrics.
    """
    url: str
    title: Optional[str] 
    meta_description: Optional[str] 
    headings: Dict[str, List[str]] 
    body_text_length: int 
    images: List[Dict[str, Optional[str]]] 
    links: Dict[str, List[str]] 
    
    # --- NEW ADVANCED SEO FIELDS ---
    canonical: Optional[str]
    robots: Optional[str]
    og_tags: Optional[str]
    schema: Optional[str]
    
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
    core_web_vitals: Optional[CoreWebVitals]
    error_message: Optional[str] = None

class CrawlState(TypedDict):  
    url: str
    status_code: int
    html_content: Optional[str]
    extracted_text: Optional[str]
    error_message: Optional[str]

class AgentState(TypedDict,CrawlState):
    url: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    crawl_result : CrawlState
    technicalAuditResult: TechnicalAuditResult
    onPageResult: OnPageResult
    marketResearchResult: MarketResearchReport
    strategyResult: StrategyResult
    optimizationResult: OptimizationResultState


class WebCrawler:
    """    
    This class is responsible for fetching and scraping web page content.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the asynchronous Firecrawl client.
        """
        if api_key is None:
            api_key = os.getenv("FIRECRAWL_API_KEY")
        
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables or parameters")
        
        self.client = Firecrawl(api_key=api_key)

    def fetch_page(self, url: str) -> CrawlState:
        """
        Asynchronously fetches and scrapes a single URL.

        Args:
            url: The URL to scrape.

        Returns:
            A CrawlResult object containing the scrape data or an error.
        """
        try:
            response = self.client.scrape(
                url,
                formats=["html", "markdown"]
            )
            
            
            # --- FIX ---
            # The response is a 'Document' object, not a 'dict'.
            # We check its type and then access its attributes.
            if isinstance(response, Document):
                return CrawlState(
                    url=url,
                    status_code=200,
                    html_content=response.html,        # Access .html attribute
                    extracted_text=response.markdown, # Access .markdown attribute
                    error_message=None
                )
            # --- END FIX ---
            else:
                # Fallback in case the response is something else unexpected
                return CrawlState(
                    url=url,
                    status_code=500,
                    html_content=None,
                    extracted_text=None,
                    error_message="Unexpected response format from scraping service."
                )

        except Exception as e:
            # Log the full exception for debugging
            return CrawlState(
                url=url,
                status_code=500,
                html_content=None,
                extracted_text=None,
                error_message=str(e)
            )
        

class TechnicalAuditor:
    """
    Performs a technical SEO audit using the Google PageSpeed Insights API (Synchronous).
    """
    def __init__(self, api_key: Optional[str] = None):
        # Fetch from env if not passed
        if api_key is None:
            api_key = os.getenv("PAGESPEED_API_KEY")
            
        if not api_key:
            raise ValueError("PageSpeed Insights API key is required.")
            
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
        self.timeout = 60.0 # Seconds

    def audit(self, url: str) -> TechnicalAuditResult:
        """Runs the audit for a given URL synchronously."""
        params = {
            'url': url,
            'key': self.api_key,
            'strategy': 'MOBILE'
        }
        error_message: Optional[str] = None

        try:
            # SYNC CHANGE: Using requests.get instead of await client.get
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            
            data = response.json()
            return self._parse_response(url, data)

        # SYNC CHANGE: Updated Exception types for 'requests' library
        except requests.exceptions.Timeout:
            error_message = f"API request timed out after {self.timeout}s. The PageSpeed API is slow."
            print(f"--- TIMEOUT ERROR --- \n{error_message}\n")
        
        except requests.exceptions.HTTPError as e:
            error_message = f"API request failed with status {e.response.status_code}: {e.response.text}"
            print(f"--- HTTP STATUS ERROR --- \n{error_message}\n")
        
        except requests.exceptions.RequestException as e:
            # Catch-all for connection errors, DNS, etc.
            error_message = f"API request failed: {type(e).__name__} - {str(e)}"
            print(f"--- REQUEST ERROR --- \n{error_message}\n")

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print("--- UNEXPECTED ERROR ---")
            traceback.print_exc()
        
        # Return a failed result object
        return TechnicalAuditResult(
            url=url, 
            performance_score=None, 
            mobile_friendly=None, 
            uses_https=None, 
            core_web_vitals=None,
            error_message=error_message
        )

    def _parse_response(self, url: str, data: dict) -> TechnicalAuditResult:
        """Parses the complex JSON response."""
        lighthouse_result = data.get('lighthouseResult', {})
        categories = lighthouse_result.get('categories', {})
        audits = lighthouse_result.get('audits', {})

        # Safe extraction with defaults
        performance_score = int(categories.get('performance', {}).get('score', 0) * 100)
        
        # 'is-on-https' returns 1 (pass) or 0 (fail)
        uses_https = audits.get('is-on-https', {}).get('score') == 1
        
        # Note: mobile-friendly is sometimes deprecated/moved, logic kept as requested
        # often found in 'viewport' or inferred from other mobile metrics now.
        mobile_friendly = True # Placeholder as specific mobile-friendly audit varies by API version
        
        # Web Vitals
        lcp = audits.get('largest-contentful-paint', {}).get('numericValue', 0) / 1000
        fid = audits.get('max-potential-fid', {}).get('numericValue', 0)
        cls = audits.get('cumulative-layout-shift', {}).get('numericValue', 0)

        vitals = CoreWebVitals(lcp=round(lcp, 2), fid=int(fid), cls=round(cls, 3))
        
        return TechnicalAuditResult(
            url=url,
            performance_score=performance_score,
            mobile_friendly=mobile_friendly,
            uses_https=uses_https,
            core_web_vitals=vitals,
            error_message=None
        )
    
class OnPageAnalyzer:
    """
    Parses HTML content to extract key on-page SEO elements, 
    including Advanced SEO tags (Canonical, OG, Robots).
    """
    def __init__(self, url: str, html_content: str):
        self.url = url
        self.base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        self.soup = BeautifulSoup(html_content, 'html.parser')

    def analyze(self) -> OnPageResult:
        """Runs all extraction methods and returns a structured result."""
        try:
            # Basic Elements
            headings = self._extract_headings()
            images = self._extract_images()
            links = self._extract_links()
            body_text = self.soup.body.get_text(separator=' ', strip=True) if self.soup.body else ""

            # Advanced SEO Elements (NEW)
            canonical = self._extract_canonical()
            robots = self._extract_robots()
            og_tags = self._extract_opengraph()
            schema = self._extract_schema()

            # We pack the new data into the result. 
            # Note: You might need to add these keys to your OnPageResult TypedDict if you want strict typing,
            # but Python dicts will accept them dynamically.
            return {
                "url": self.url,
                "title": self._extract_title(),
                "meta_description": self._extract_meta_description(),
                "headings": headings,
                "body_text_length": len(body_text.split()),
                "images": images,
                "links": links,
                # New Advanced Fields
                "canonical": canonical,
                "robots": robots,
                "og_tags": og_tags,
                "schema": schema,
                "error_message": None
            }
        except Exception as e:
            return {"url": self.url, "error_message": f"Parsing Error: {str(e)}"}

    def _extract_title(self) -> Optional[str]:
        return self.soup.title.string.strip() if self.soup.title else None

    def _extract_meta_description(self) -> Optional[str]:
        meta = self.soup.find('meta', attrs={'name': 'description'})
        return meta.get('content').strip() if meta else None

    def _extract_headings(self) -> Dict[str, List[str]]:
        headings = {}
        for i in range(1, 4): # H1 to H3
            tag = f'h{i}'
            found_tags = [h.get_text(strip=True) for h in self.soup.find_all(tag)]
            if found_tags: headings[tag] = found_tags
        return headings

    def _extract_images(self) -> List[Dict[str, Optional[str]]]:
        return [{'src': i.get('src'), 'alt': i.get('alt')} for i in self.soup.find_all('img')]

    def _extract_links(self) -> Dict[str, List[str]]:
        links = {'internal': [], 'external': []}
        for a in self.soup.find_all('a', href=True):
            href = urljoin(self.base_url, a['href'])
            if self.base_url in href: links['internal'].append(href)
            else: links['external'].append(href)
        return links

    # --- NEW ADVANCED METHODS ---

    def _extract_canonical(self) -> str:
        link = self.soup.find('link', rel='canonical')
        return link.get('href') if link else "Missing"

    def _extract_robots(self) -> str:
        meta = self.soup.find('meta', attrs={'name': 'robots'})
        return meta.get('content') if meta else "No specific directives (Index/Follow assumed)"

    def _extract_opengraph(self) -> str:
        # Check for og:title, og:description, og:image
        og_props = ['og:title', 'og:description', 'og:image']
        found = []
        for prop in og_props:
            if self.soup.find('meta', property=prop):
                found.append(prop)
        
        if len(found) == 3: return "Perfect (Title, Desc, Image found)"
        if found: return f"Partial ({', '.join(found)})"
        return "Missing"

    def _extract_schema(self) -> str:
        # Check for JSON-LD schema
        schema = self.soup.find('script', type='application/ld+json')
        return "Detected (JSON-LD)" if schema else "Missing"
    
class MarketResearcher:
    """
    Performs keyword analysis, competitor research, and content gap analysis (Synchronous).
    """
    def __init__(self, page_content: str, page_title: str):
        self.page_content = page_content
        self.page_title = page_title
        
        # LLM Setup
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2, # Low temperature for factual, analytical reporting
        )
        
        # Search Tool Setup (Requires TAVILY_API_KEY env var)
        self.search_tool = TavilySearchResults(max_results=3)

    def research(self) -> MarketResearchReport:
        """Runs the full research pipeline synchronously."""
        try:
            print("   > Identifying keywords...")
            # 1. Identify keywords
            keyword_data = self._identify_keywords()
            
            # Safe check: ensure we have a dictionary and the key exists
            primary_kw = keyword_data.get("primary_keyword")
            
            if not primary_kw:
                raise ValueError("Could not determine a primary keyword.")

            print(f"   > Analyzing competitors for: {primary_kw}...")

            # 2. Find competitors
            competitors = self._analyze_competitors(primary_kw)
            
            # Extract URLs and create snippets
            # Check if 'url' and 'content' exist safely
            competitor_urls = [res.get('url') for res in competitors if isinstance(res, dict) and res.get('url')]
            
            competitor_snippets = "\n".join(
                [f"URL: {res.get('url')}\nContent: {res.get('content')}" for res in competitors if isinstance(res, dict)]
            )

            print("   > Calculating content gaps...")
            # 3. Find content gaps
            gap_data = self._find_content_gaps(competitor_snippets)

            return MarketResearchReport(
                keyword_analysis=keyword_data,
                competitor_urls=competitor_urls,
                content_gap_analysis=gap_data,
                error_message=None
            )

        except Exception as e:
            # Return empty/safe structure on failure
            return MarketResearchReport(
                keyword_analysis=None,
                competitor_urls=[],
                content_gap_analysis=None,
                error_message=str(e)
            )

    def _identify_keywords(self) -> dict:
        """Extracts keywords using Gemini and returns a DICTIONARY."""
        prompt = ChatPromptTemplate.from_template(
            "Analyze the webpage content. Identify 1 primary keyword and 3-5 secondary keywords.\n\nTitle: {title}\n\nContent: {content}"
        )
        # InternalKeywordSchema must be defined in your file
        chain = prompt | self.llm.with_structured_output(InternalKeywordSchema)
        
        result = chain.invoke({
            "title": self.page_title,
            "content": self.page_content[:10000],
        })
        
        # --- CRITICAL FIX: Convert Pydantic Object to Dictionary ---
        if hasattr(result, 'model_dump'): return result.model_dump() # Pydantic v2
        if hasattr(result, 'dict'): return result.dict() # Pydantic v1
        return result

    def _analyze_competitors(self, primary_keyword: str) -> List[Dict[str, Any]]:
        """Uses Tavily Search to find top competing pages."""
        # Tavily returns a list of dicts naturally, so this is usually fine
        return self.search_tool.invoke(f"top blog posts about {primary_keyword}")

    def _find_content_gaps(self, competitor_snippets: str) -> dict:
        """Finds missing topics and returns a DICTIONARY."""
        prompt = ChatPromptTemplate.from_template(
            "Compare my content with competitor snippets. What are they covering that I am missing? \n\nMy Content: {my_content}\n\nCompetitor Content:\n{competitor_content}"
        )
        # InternalGapSchema must be defined in your file
        chain = prompt | self.llm.with_structured_output(InternalGapSchema)
        
        result = chain.invoke({
            "my_content": self.page_content[:5000],
            "competitor_content": competitor_snippets,
        })
        
        # --- CRITICAL FIX: Convert Pydantic Object to Dictionary ---
        if hasattr(result, 'model_dump'): return result.model_dump() # Pydantic v2
        if hasattr(result, 'dict'): return result.dict() # Pydantic v1
        return result

    def _identify_keywords(self) -> KeywordReport:
        """Extracts keywords using Gemini."""
        prompt = ChatPromptTemplate.from_template(
            "Analyze the webpage content. Identify 1 primary keyword and 3-5 secondary keywords.\n\nTitle: {title}\n\nContent: {content}"
        )
        # We use the Pydantic class for structure, but immediately .dict() it
        chain = prompt | self.llm.with_structured_output(InternalKeywordSchema)
        
        result = chain.invoke({
            "title": self.page_title,
            "content": self.page_content[:10000], # Gemini 1.5 handles large context easily
        })
        # Convert Pydantic object to simple Dict
        return result

    def _analyze_competitors(self, primary_keyword: str) -> List[Dict[str, Any]]:
        """Uses Tavily Search to find top competing pages."""
        # Synchronous invoke
        return self.search_tool.invoke(f"top blog posts about {primary_keyword}")

    def _find_content_gaps(self, competitor_snippets: str) -> ContentGapReport:
        """Finds missing topics."""
        prompt = ChatPromptTemplate.from_template(
            "Compare my content with competitor snippets. What are they covering that I am missing? \n\nMy Content: {my_content}\n\nCompetitor Content:\n{competitor_content}"
        )
        chain = prompt | self.llm.with_structured_output(InternalGapSchema)
        
        result = chain.invoke({
            "my_content": self.page_content[:5000],
            "competitor_content": competitor_snippets,
        })
        return result
    
class SeoStrategist:
    """
    Synthesizes audit data into an actionable SEO strategy using Gemini (Sync).
    """
    def __init__(self, onpage_data: dict, technical_data: dict, research_data: dict):
        self.onpage_data = onpage_data
        self.technical_data = technical_data
        self.research_data = research_data
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

    def generate_strategy(self) -> dict:
        """Generates the strategy synchronously."""
        formatted_data = self._format_data_for_prompt()
        
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert SEO consultant. Review the audit data below and create a prioritized list of action items.
            
            Be specific. Don't say "Fix LCP". Say "Optimize hero image size to reduce LCP".
            
            AUDIT DATA:
            {audit_data}
            """
        )
        
        # Structure the output
        chain = prompt | self.llm.with_structured_output(SeoStrategy)
        
        try:
            # Sync Invoke
            result = chain.invoke({"audit_data": formatted_data})
            
            # Convert Pydantic -> Dict for LangGraph State compatibility
            # .dict() is used in Pydantic v1 (LangChain default), .model_dump() in v2
            if hasattr(result, 'model_dump'):
                return result.model_dump() # Pydantic v2
            if hasattr(result, 'dict'):
                return result.dict() # Pydantic v1 (You were missing the parentheses here)
            
            # Fallback if neither exists (unlikely if result is SeoStrategy)
            return result
            
        except Exception as e:
            return {
                "recommendations": [
                    {
                        "priority": "High",
                        "category": "System",
                        "recommendation": "Strategy Generation Failed",
                        "justification": str(e)
                    }
                ]
            }

    def _format_data_for_prompt(self) -> str:
        """Formats the input dicts into a string, handling potential serialization errors."""
        def safe_json(data):
            try:
                return json.dumps(data, indent=2, default=str)
            except:
                return str(data)

        return f"""
        === ON-PAGE ANALYSIS ===
        {safe_json(self.onpage_data)}

        === TECHNICAL AUDIT ===
        {safe_json(self.technical_data)}

        === MARKET RESEARCH ===
        {safe_json(self.research_data)}
        """

class SeoOptimizer:
    """
    Executes SEO strategy by rewriting content and generating new sections (Synchronous).
    """
    def __init__(self, strategy: Dict, onpage_data: Dict, research_data: Dict):
        # Safely extract recommendations list
        self.strategy = strategy.get('recommendations', []) if strategy else []
        self.onpage_data = onpage_data or {}
        self.research_data = research_data or {}
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    def apply_optimizations(self) -> Dict:
        """Runs the optimization pipeline synchronously."""
        try:
            # Filter recommendations
            title_recs = [r for r in self.strategy if r.get('category') in ['Title Tag', 'Meta Description', 'On-Page']]
            content_recs = [r for r in self.strategy if r.get('category') == 'Content']

            optimized_meta = None
            new_sections = []

            # 1. Rewrite Title & Meta (if needed)
            # We look for recommendations regarding title/meta
            # Logic: If category matches OR the text mentions "title"
            if title_recs: 
                print("   > Optimizing Title & Meta Description...")
                optimized_meta = self._rewrite_title_and_meta(title_recs)

            # 2. Generate New Sections (Limit to top 2 to save time/tokens)
            # We look for "add", "gap", or "create" in the recommendation text
            gap_recs = [
                r for r in content_recs 
                if any(x in r.get('recommendation', '').lower() for x in ['add', 'gap', 'create', 'write'])
            ][:2]
            
            if gap_recs:
                print(f"   > Generating {len(gap_recs)} new content sections...")
                new_sections = self._generate_new_sections(gap_recs)

            return {
                "optimized_title_meta": optimized_meta,
                "new_sections": new_sections,
                "error_message": None
            }

        except Exception as e:
            return {
                "optimized_title_meta": None, 
                "new_sections": [], 
                "error_message": str(e)
            }

    def _rewrite_title_and_meta(self, recommendations: List[Dict]) -> Dict:
        """Rewrites title/meta."""
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert SEO copywriter. Rewrite the title and meta description.
            
            Original Title: {original_title}
            Original Meta: {original_meta}
            Target Keyword: {keyword}
            
            Directives:
            {recommendations}
            """
        )
        chain = prompt | self.llm.with_structured_output(OptimizedTitleMeta)
        
        # Extract Keyword safely
        kw_data = self.research_data.get('keyword_analysis')
        # Handle if kw_data is None or empty
        if not kw_data: 
            kw_data = {}
            
        primary_kw = kw_data.get('primary_keyword', 'N/A')
        
        rec_text = "\n".join([f"- {r.get('recommendation')}" for r in recommendations])

        result = chain.invoke({
            "original_title": self.onpage_data.get('title', ''),
            "original_meta": self.onpage_data.get('meta_description', ''),
            "keyword": primary_kw,
            "recommendations": rec_text
        })
        
        # --- FIX: Support Pydantic V2 (.model_dump) and V1 (.dict) ---
        if hasattr(result, 'model_dump'): return result.model_dump()
        if hasattr(result, 'dict'): return result.dict()
        return result

    def _generate_new_sections(self, recommendations: List[Dict]) -> List[Dict]:
        """Generates new content paragraphs."""
        # Extract Keyword safely
        kw_data = self.research_data.get('keyword_analysis')
        if not kw_data: 
            kw_data = {}
            
        primary_kw = kw_data.get('primary_keyword', 'N/A')
        
        generated = []
        
        prompt = ChatPromptTemplate.from_template(
            """
            Write a new website content section (100-150 words) to fill a content gap.
            Topic: "{topic}"
            Target Keyword: "{keyword}"
            """
        )
        chain = prompt | self.llm.with_structured_output(GeneratedContentSection)

        for rec in recommendations:
            topic = rec.get('recommendation', 'General Topic')
            try:
                res = chain.invoke({"topic": topic, "keyword": primary_kw})
                
                # --- FIX: Support Pydantic V2 (.model_dump) and V1 (.dict) ---
                if hasattr(res, 'model_dump'):
                    res_dict = res.model_dump()
                elif hasattr(res, 'dict'):
                    res_dict = res.dict()
                else:
                    res_dict = res
                    
                generated.append(res_dict)
            except Exception as e:
                print(f"Failed to generate section for {topic}: {e}")
                
        return generated
    
def crawl_node(state: AgentState) -> AgentState:
    """
    Executes the crawling logic and updates the state with results.
    """
    print(f"--- CRAWLING URL: {state['url']} ---")
    
    # 1. Extract the target URL from the state
    target_url = state["url"]
    
    crawler = WebCrawler()
    
    result = crawler.fetch_page(target_url)
    
    if result["status_code"] == 200:
        summary_msg = f"Successfully crawled {target_url}. Content is ready for audit."
    else:
        summary_msg = f"Failed to crawl {target_url}. Error: {result['error_message']}"
        
    return {
        "crawl_result": result,
        "messages": [SystemMessage(content=summary_msg)]
    }

def technical_audit_node(state: AgentState) -> AgentState:
    """
    Node responsible for running the PageSpeed Insights audit.
    """
    print(f"--- RUNNING TECHNICAL AUDIT: {state['url']} ---")
    
    target_url = state["url"]
    
    # 1. Initialize Auditor
    # Ensure 'PAGE_SPEED_INSIGHTS_API_KEY' is in your .env file
    auditor = TechnicalAuditor() 
    
    # 2. Run Synchronous Audit
    audit_result = auditor.audit(target_url)
    
    # 3. Prepare System Message
    # This informs the LLM (in future nodes) what happened.
    if audit_result['error_message']:
        msg_content = f"Technical audit failed: {audit_result['error_message']}"
    else:
        msg_content = (
            f"Technical audit complete. "
            f"Performance Score: {audit_result['performance_score']}/100. "
            f"LCP: {audit_result['core_web_vitals']['lcp']}s. "
        )

    # 4. Update State
    return {
        "technicalAuditResult": audit_result,
        "messages": [SystemMessage(content=msg_content)]
    }

def onpage_analyzer_node(state: AgentState) -> dict:
    """
    Node that analyzes the HTML content for On-Page SEO elements, 
    including Advanced SEO metrics.
    """
    print(f"--- RUNNING ON-PAGE ANALYSIS: {state['url']} ---")
    
    # 1. Get the HTML from the previous Crawl Node
    crawl_data = state.get("crawl_result")
    
    # Safety Check: Did the crawler fail?
    if not crawl_data or not crawl_data.get("html_content"):
        return {
            "onPageResult": {"error_message": "Skipping analysis: No HTML content available from crawl."}
        }

    html_content = crawl_data["html_content"]
    url = state["url"]

    # 2. Initialize the user's Analyzer Class
    analyzer = OnPageAnalyzer(url, html_content)
    
    # 3. Run the Analysis
    results = analyzer.analyze()

    # 4. Create a Summary for the LLM
    # We summarize key issues here so the LLM doesn't have to process raw JSON arrays
    
    # Basic Metrics
    img_missing_alt = len([img for img in results['images'] if not img.get('alt')])
    h1_count = len(results.get('headings', {}).get('h1', []))
    title_len = len(results['title']) if results.get('title') else 0
    
    # Advanced Metrics (New)
    og_status = results.get('og_tags', 'Missing')
    schema_status = results.get('schema', 'Missing')
    robots_status = results.get('robots', 'Missing')
    
    # Check if canonical exists and is not the literal string "Missing"
    canonical_val = results.get('canonical')
    canonical_status = "Present" if canonical_val and canonical_val != "Missing" else "Missing"
    
    summary_msg = (
        f"On-Page Analysis Complete.\n"
        f"- Title Length: {title_len} chars\n"
        f"- H1 Tags Found: {h1_count}\n"
        f"- Images without Alt Text: {img_missing_alt}\n"
        f"- Internal Links: {len(results.get('links', {}).get('internal', []))}\n"
        f"- External Links: {len(results.get('links', {}).get('external', []))}\n"
        f"- Advanced SEO: Canonical ({canonical_status}), Robots ({robots_status}), OG Tags ({og_status}), Schema ({schema_status})"
    )

    # 5. Return the state update
    return {
        "onPageResult": results,
        "messages": [SystemMessage(content=summary_msg)]
    }

def market_research_node(state: AgentState) -> AgentState:
    """
    Node that runs competitor analysis and keyword research.
    """
    print(f"--- RUNNING MARKET RESEARCH: {state['url']} ---")
    
    # 1. Get prerequisites
    crawl_data = state.get("crawl_result", {})
    on_page_data = state.get("onPageResult", {})
    
    html_content = crawl_data.get("html_content")
    
    # Safety check
    if not html_content:
        return {
            "marketResearchResult": {"error_message": "Skipping research: No HTML content available."}
        }

    soup = BeautifulSoup(html_content, 'html.parser')
    page_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
    
    page_title = on_page_data.get("title", state["url"])

    researcher = MarketResearcher(page_content=page_text, page_title=page_title)
    results = researcher.research()

    # 4. Return State Update
    return {
        "marketResearchResult": results
    }

def strategy_node(state: AgentState) -> AgentState:
    """
    Node that runs the SeoStrategist.
    """
    print(f"--- GENERATING STRATEGY: {state['url']} ---")
    
    # 1. Extract inputs from State
    onpage = state.get("onPageResult", {})
    tech = state.get("technicalAuditResult", {})
    market = state.get("marketResearchResult", {})
    
    # 2. Run Strategist
    strategist = SeoStrategist(
        onpage_data=onpage,
        technical_data=tech,
        research_data=market
    )
    
    result = strategist.generate_strategy()
    
    # 3. Return update
    return {"strategyResult": result}

def optimization_node(state: AgentState) -> AgentState:
    """
    Node that runs the SeoOptimizer.
    """
    print(f"--- RUNNING CONTENT OPTIMIZER: {state['url']} ---")
    
    # 1. Extract inputs
    strategy = state.get("strategyResult", {})
    onpage = state.get("onPageResult", {})
    research = state.get("marketResearchResult", {})
    
    # 2. Run Optimizer
    optimizer = SeoOptimizer(strategy, onpage, research)
    results = optimizer.apply_optimizations()
    
    # 3. Update State
    return {"optimizationResult": results}

# ==========================================
# 1. LONG-FORM PROFESSIONAL TEMPLATE (NO ICONS)
# ==========================================
SEO_REPORT_TEMPLATE = """
# COMPREHENSIVE SEO AUDIT REPORT
**Target Website:** {url}
**Audit Date:** {date}

---

# TABLE OF CONTENTS
1. Executive Summary
2. Scorecard Overview
3. Search Engine Visibility
4. On-Page SEO Analysis
    4.1 Title Tag Optimization
    4.2 Meta Description Analysis
    4.3 Keyword Usage
5. Content Structure & Hierarchy
    5.1 Heading Tags (H1-H6)
    5.2 Content Quality & Gaps
6. Technical Infrastructure
    6.1 Canonicalization
    6.2 Robots & Indexing
    6.3 Schema Markup
7. Site Performance & Core Web Vitals
8. Security & Accessibility
9. Strategic Action Plan

---

# 1. EXECUTIVE SUMMARY

This document provides an in-depth technical and content audit of {url}. The purpose of this audit is to identify barriers preventing the site from ranking higher in search engines and to uncover opportunities for organic growth.

**Current Health Assessment**
The website is currently performing at a score of **{perf_score}/100**.

Based on our analysis, the website demonstrates strengths in certain areas but requires immediate attention regarding critical technical SEO factors and content optimization. The following sections detail every aspect of the site's performance, contrasting current metrics against industry standards and Google's Core Web Vitals benchmarks.

---

# 2. SCORECARD OVERVIEW

We have categorized our findings into three distinct levels of urgency. This breakdown helps prioritize the remediation efforts required.

| Category | Findings Count | Definition |
| :--- | :--- | :--- |
| **Good Results** | {good_count} | Items that meet or exceed industry best practices. No action required. |
| **Recommended Improvements** | {rec_count} | Items that are not broken but could be optimized for better performance. |
| **Critical Issues** | {critical_count} | Items that are actively hurting your rankings or user experience. Immediate fix required. |

---

# 3. SEARCH ENGINE VISIBILITY

This section visualizes how your website appears to a user on a Search Engine Results Page (SERP). The click-through rate (CTR) of your website is heavily influenced by the attractiveness and relevance of this snippet.

**Snippet Preview:**
> **{seo_title}**
> {url}
> {seo_description}

**Analysis of Snippet:**
If the title or description above appears cut off, generic, or missing, it indicates a significant missed opportunity. A well-crafted snippet acts as an advertisement for your content, enticing users to choose your link over competitors.

---

# 4. ON-PAGE SEO ANALYSIS

On-page SEO refers to the practice of optimizing individual web pages in order to rank higher and earn more relevant traffic in search engines.

## 4.1 Title Tag Optimization

**Concept Definition**
The Title Tag is an HTML element that specifies the title of a web page. It is displayed on search engine results pages (SERPs) as the clickable headline for a given result.

**Why It Matters**
The title tag is widely considered one of the most important on-page SEO elements. It gives search engines a high-level overview of the page content and is the first interaction a user has with your brand in search results.

**Audit Findings**
* **Status:** {title_status}
* **Character Length:** {title_len} characters (Optimal: 50-60 characters)
* **Current Content:** "{seo_title}"

**Recommendations**
Ensure your title tag is unique for every page. It should place the most important keyword near the beginning of the string. If the current length is under 30 characters, it is too thin. If it is over 60 characters, Google will truncate it.

## 4.2 Meta Description Analysis

**Concept Definition**
The meta description is an attribute that provides a brief summary of a web page. Search engines often display the meta description in search results, which can influence click-through rates.

**Why It Matters**
While meta descriptions are not a direct ranking factor, they are a primary driver of user behavior. A compelling description acts as a "pitch" to the searcher. If left empty, Google will pull random text from the page, which may not be flattering or relevant.

**Audit Findings**
* **Status:** {meta_status}
* **Character Length:** {meta_len} characters (Optimal: 150-160 characters)
* **Current Content:** "{seo_description}"

## 4.3 Keyword Usage

**Concept Definition**
Keywords are the ideas and topics that define what your content is about. In terms of SEO, they're the words and phrases that searchers enter into search engines, also called "search queries."

**Audit Findings**
* **Primary Keyword Identified:** {keyword_status}

**Analysis**
We analyzed the content to see if a primary keyword is naturally integrated into the Title, H1, and first 100 words of the body content. If the primary keyword is listed as "N/A" or "Missing," search engines may struggle to understand the core topic of this page.

---

# 5. CONTENT STRUCTURE & HIERARCHY

Search engines use heading tags to understand the structure and hierarchy of your content.

## 5.1 Heading Tags (H1-H6)

**Concept Definition**
Heading tags are used to communicate the organization of the content on the page. The H1 tag should define the main topic, while H2s through H6s should be used to define sub-topics.

**Why It Matters**
Proper usage of heading tags allows search engine crawlers to navigate your content efficiently. It also improves accessibility for screen readers. A page should strictly have only one H1 tag.

**Audit Findings**
* **H1 Tag Status:** {h1_status}
* **Current H1:** "{h1_tag}"

**Sub-heading Distribution (H2 Tags):**
{h2_list}

**Analysis**
Review the list of H2 tags above. Do they outline a logical flow of information? Do they contain secondary keywords? If the list is empty, the content may appear as a "wall of text" to Google, which is difficult to index accurately.

## 5.2 Content Quality & Gaps

**Concept Definition**
Content Gap Analysis involves comparing your existing content against that of your top competitors to identify topics you have missed.

**Market Intelligence Findings**
We analyzed the top ranking competitors for your niche.
* **Competitor Count:** {comp_count}

**Identified Content Gaps:**
The following topics are covered by your competitors but appear to be missing or under-represented on your page:
{gap_list}

**Recommendation**
To establish topical authority, we recommend expanding your content to include sections dedicated to these missing topics. This demonstrates to Google that your page is the most comprehensive resource available.

---

# 6. TECHNICAL INFRASTRUCTURE

Technical SEO refers to website and server optimizations that help search engine spiders crawl and index your site more effectively.

## 6.1 Canonicalization

**Concept Definition**
A canonical tag (rel="canonical") is a snippet of HTML code that defines the main version for duplicate, near-duplicate, and similar pages.

**Audit Findings**
* **Status:** {canonical_status}

**Analysis**
If this is "Missing," your site is at risk of duplicate content issues, especially if you use URL parameters (like tracking codes) in marketing campaigns.

## 6.2 Robots & Indexing

**Concept Definition**
The meta robots tag tells search engines whether they are allowed to index this specific page and whether they should follow the links upon it.

**Audit Findings**
* **Directives Found:** {robots_status}

## 6.3 Schema Markup

**Concept Definition**
Schema markup is code (semantic vocabulary) that you put on your website to help the search engines return more informative results for users.

**Audit Findings**
* **Status:** {schema_status}

**Analysis**
If schema is detected, it increases the likelihood of rich snippets (stars, images, FAQ boxes) appearing in search results. If missing, you are losing real estate on the results page.

---

# 7. SITE PERFORMANCE & CORE WEB VITALS

Google has officially made page speed a ranking factor. This section analyzes the Core Web Vitals, which are a set of specific factors that Google considers important in a webpage's overall user experience.

## Performance Metrics

| Metric | Value | Assessment |
| :--- | :--- | :--- |
| **Overall Performance Score** | {perf_score}/100 | General Health Indicator |
| **Largest Contentful Paint (LCP)** | {lcp}s | Measures Loading Performance |
| **Cumulative Layout Shift (CLS)** | {cls} | Measures Visual Stability |
| **Response Time** | {response_time}s | Server Latency |
| **Page Size** | {html_size_kb} KB | Code Heaviness |

## Detailed Analysis

**Largest Contentful Paint (LCP)**
LCP measures how long it takes for the largest content element (usually the hero image or main text) to become visible.
* **Your Result:** {lcp} seconds.
* **Benchmark:** Google requires this to be under 2.5 seconds.
* **Impact:** Slow LCP causes high bounce rates as users become frustrated waiting for the main content to appear.

**Optimization Opportunities**
* **JavaScript Minification Status:** {js_min_status}
* **CSS Minification Status:** {css_min_status}
* **Total Requests:** {request_count}

---

# 8. SECURITY & ACCESSIBILITY

Website security is a prerequisite for ranking. Google prioritizes the safety of its users.

## Security Audit

* **HTTPS (SSL Certificate):** {https_status}
* **Directory Listing:** {dir_listing_status}
* **Malware Status:** {malware_status}

**Analysis**
HTTPS (Hypertext Transfer Protocol Secure) is an internet communication protocol that protects the integrity and confidentiality of data between the user's computer and the site. If your status is "Insecure," browsers like Chrome will mark your site as "Not Secure," significantly hurting trust and conversion rates.

## Image Accessibility

* **Total Images:** {img_count}
* **Images Missing Alt Text:** {img_missing_alt}

**Why it Matters**
Alt text is a written description of an image. Screen readers use this to describe images to visually impaired users. Furthermore, search engines use this text to understand what the image shows, helping you rank in Google Image Search.

---

# 9. STRATEGIC ACTION PLAN

Based on the comprehensive data collected above, we have developed a prioritized roadmap. These recommendations are ordered by impact (High to Low). Executing the High Priority items will yield the fastest results.

## High Priority (Critical Fixes)
These items are likely preventing your site from ranking or causing penalty risks.

{action_plan_high}

## Medium Priority (Optimization)
These items will help improve your keyword rankings and click-through rates.

{action_plan_medium}

## Low Priority (Maintenance)
These items are best practices for long-term health.

{action_plan_low}

---
*End of Report | Generated by AI SEO Agent*
"""
from datetime import datetime

def report_node(state: AgentState) -> dict:
    """
    Fills the LONG-FORM template using data from all agents.
    """
    print(f"--- GENERATING COMPREHENSIVE REPORT FOR: {state['url']} ---")

    # 1. Extract Data
    crawl = state.get("crawl_result", {})
    audit = state.get("technicalAuditResult", {})
    onpage = state.get("onPageResult", {})
    market = state.get("marketResearchResult", {})
    strategy = state.get("strategyResult", {})
    opt = state.get("optimizationResult", {})
    url = state.get("url")

    if crawl.get("error_message"):
        return {"messages": [SystemMessage(content=f"Report Failed: {crawl['error_message']}")]}

    # --- Helper: Safe Get ---
    def safe_get(obj, key, default="N/A"):
        if not obj: return default
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    # 2. Prepare Variables
    
    # Overview
    perf_score = safe_get(audit, 'performance_score', 0)
    
    # Basic SEO
    title = safe_get(onpage, 'title', 'Missing')
    meta = safe_get(onpage, 'meta_description', 'Missing')
    
    # Headings
    headings = safe_get(onpage, 'headings', {})
    h1s = safe_get(headings, 'h1', [])
    h1_tag = h1s[0] if h1s else "No H1 Tag Found"
    
    h2s = safe_get(headings, 'h2', [])
    h2_list_str = "\n".join([f"- {h}" for h in h2s[:15]]) if h2s else "No H2 tags found."

    # Images & Links
    images = safe_get(onpage, 'images', [])
    img_missing_alt = len([i for i in images if not safe_get(i, 'alt')])
    
    # Tech
    core_vitals = safe_get(audit, 'core_web_vitals', {})
    lcp = safe_get(core_vitals, 'lcp', 0)
    cls = safe_get(core_vitals, 'cls', 0)
    
    # Market
    kw_data = safe_get(market, 'keyword_analysis', {})
    primary_kw = safe_get(kw_data, 'primary_keyword', 'N/A')
    competitors = safe_get(market, 'competitor_urls', [])
    gaps = safe_get(safe_get(market, 'content_gap_analysis', {}), 'suggested_topics', [])
    gap_list_str = "\n".join([f"- {gap}" for gap in gaps]) if gaps else "No specific content gaps detected."

    # Strategy Separation (High/Med/Low)
    recs = safe_get(strategy, 'recommendations', [])
    if recs and isinstance(recs[0], object) and hasattr(recs[0], 'dict'):
        recs = [r.dict() for r in recs]
        
    high_recs = [f"**{r.get('category')}:** {r.get('recommendation')} ({r.get('justification')})" for r in recs if r.get('priority', '').lower() == 'high']
    med_recs = [f"**{r.get('category')}:** {r.get('recommendation')} ({r.get('justification')})" for r in recs if r.get('priority', '').lower() == 'medium']
    low_recs = [f"**{r.get('category')}:** {r.get('recommendation')} ({r.get('justification')})" for r in recs if r.get('priority', '').lower() == 'low']

    high_str = "\n\n".join(high_recs) if high_recs else "No critical high-priority issues found."
    med_str = "\n\n".join(med_recs) if med_recs else "No medium-priority optimization suggested."
    low_str = "\n\n".join(low_recs) if low_recs else "No low-priority maintenance items."

    # 3. Construct Context for LLM
    system_prompt = """You are a Senior Technical SEO Consultant writing a white-paper style audit report.
    
    Your goal is to populate the provided Long-Form Template.
    
    INSTRUCTIONS:
    1. **Tone:** Professional, authoritative, and educational.
    2. **Length:** Do not summarize. Write full, detailed paragraphs for the 'Analysis' sections.
    3. **Icons:** DO NOT use emojis or icons (like ✅, ❌, 💡). Use text-based status indicators (e.g., "Pass", "Fail", "Critical", "Good").
    4. **Data:** Use the Context Data provided to fill the variables.
    5. **Scorecard:**
       - 'Critical Issues': Count items where LCP > 2.5s, HTTPS is Insecure, or Title is Missing.
       - 'Good Results': Count items where Score > 80, LCP < 2.5s, Title exists.
    """

    user_message = f"""
    TARGET TEMPLATE:
    {SEO_REPORT_TEMPLATE}

    ---
    CONTEXT DATA:
    
    [METADATA]
    - URL: {url}
    - Date: {datetime.now().strftime("%B %d, %Y")}
    
    [TECHNICAL]
    - Score: {perf_score}
    - LCP: {lcp}
    - CLS: {cls}
    - HTTPS: {safe_get(audit, 'uses_https')}
    - Mobile Friendly: {safe_get(audit, 'mobile_friendly')}
    
    [ON-PAGE]
    - Title: {title}
    - Meta: {meta}
    - H1: {h1_tag}
    - H2 Count: {len(h2s)}
    - Image Count: {len(images)}
    - Missing Alt: {img_missing_alt}
    
    [MARKET]
    - Keyword: {primary_kw}
    - Competitor Count: {len(competitors)}
    - Gaps: {gap_list_str}
    
    [STRATEGY STRINGS]
    High Priority: {high_str}
    Medium Priority: {med_str}
    Low Priority: {low_str}
    """

    # 4. Invoke LLM
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    return {"messages": [response]}

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("crawler", crawl_node)
workflow.add_node("auditor", technical_audit_node) # Can run parallel to Crawler
workflow.add_node("onpage_analyzer", onpage_analyzer_node)
workflow.add_node("market_researcher", market_research_node)
workflow.add_node("strategist", strategy_node)
workflow.add_node("optimizer", optimization_node)
workflow.add_node("reporter", report_node)


# Define Edges

workflow.add_edge(START, "auditor")
workflow.add_edge(START, "crawler")
workflow.add_edge("crawler", "onpage_analyzer")
workflow.add_edge("onpage_analyzer", "market_researcher")
workflow.add_edge("auditor", "reporter")
workflow.add_edge("market_researcher", "strategist")
workflow.add_edge("strategist", "optimizer")
workflow.add_edge("optimizer", "reporter")
workflow.add_edge("reporter", END)

app = workflow.compile()

# --- HOW TO RUN IT ---
if __name__ == "__main__":
    
    # Define the initial state
    initial_state = {
        "url": "https://www.greenstechnologys.com/", # Replace with the URL you want to audit
        "messages": []
    }

    print("--- STARTING SEO AGENT ---")
    
    # Stream the output
    for event in app.stream(initial_state):
        for key, value in event.items():
            print(f"\nFinished Node: {key}")
            
            # 1. Handle Final Report
            if key == "reporter":
                print("\n" + "="*40)
                print("FINAL GEMINI REPORT")
                print("="*40 + "\n")
                # access messages safely
                msgs = value.get("messages", [])
                if msgs:
                    print(msgs[-1].content)

            # 2. Handle Strategy Plan (FIXED)
            if key == "strategist":
                print("\n" + "="*40)
                print("STRATEGIC ACTION PLAN")
                print("="*40)
                
                data = value["strategyResult"]
                
                # Helper to get recommendations list whether data is Dict or Pydantic Object
                if isinstance(data, dict):
                    recs = data.get("recommendations", [])
                else:
                    # If it's still a Pydantic object, use getattr
                    recs = getattr(data, "recommendations", [])
                    # If recommendations inside are also objects, convert to dict for printing
                    if recs and not isinstance(recs[0], dict) and hasattr(recs[0], 'dict'):
                         recs = [r.dict() for r in recs]

                for i, rec in enumerate(recs, 1):
                    # handle accessing keys
                    prio = rec.get('priority', 'N/A').upper()
                    cat = rec.get('category', 'General')
                    text = rec.get('recommendation', 'No text')
                    print(f"{i}. [{prio}] {cat}: {text}")