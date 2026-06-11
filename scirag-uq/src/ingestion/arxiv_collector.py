import os
import json
import urllib.request
import xml.etree.ElementTree as ET
import time
from typing import List, Dict

# arXiv API namespace
OAI = "{http://www.w3.org/2005/Atom}"

class ArxivCollector:
    """
    Searches arXiv, downloads PDFs, and saves metadata.
    """
    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = base_dir
        self.categories = ["robotics", "rag", "hallucination", "scientific_qa", "antbot"]
        self.metadata_file = os.path.join(self.base_dir, "metadata.json")
        self.metadata = []
        
        for cat in self.categories:
            os.makedirs(os.path.join(self.base_dir, cat), exist_ok=True)
            
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = []

    def fetch_papers(self, query: str, category: str, max_results: int = 20):
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}")
            
        print(f"Fetching {max_results} papers for query: '{query}' -> {category}")
        
        # arXiv API
        search_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        
        try:
            data = urllib.request.urlopen(url).read()
            root = ET.fromstring(data)
            entries = root.findall(f"{OAI}entry")
            
            for entry in entries:
                title = entry.find(f"{OAI}title").text.strip().replace("\n", "")
                authors = [a.find(f"{OAI}name").text for a in entry.findall(f"{OAI}author")]
                published = entry.find(f"{OAI}published").text
                year = int(published.split("-")[0])
                
                # Find PDF link
                pdf_url = None
                for link in entry.findall(f"{OAI}link"):
                    if link.attrib.get("title") == "pdf":
                        pdf_url = link.attrib.get("href")
                        break
                        
                if not pdf_url:
                    continue
                    
                entry_id = entry.find(f"{OAI}id").text.split("/")[-1]
                pdf_path = os.path.join(self.base_dir, category, f"{entry_id}.pdf")
                
                # Check if already downloaded
                if not os.path.exists(pdf_path):
                    print(f"Downloading: {title[:50]}...")
                    try:
                        urllib.request.urlretrieve(pdf_url + ".pdf", pdf_path)
                        time.sleep(3) # Rate limit respect
                    except Exception as e:
                        print(f"Failed to download {pdf_url}: {e}")
                        continue
                else:
                    print(f"Already exists: {title[:50]}...")
                
                meta = {
                    "id": entry_id,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "category": category,
                    "pdf_path": pdf_path,
                    "url": entry.find(f"{OAI}id").text
                }
                
                # Avoid duplicates in metadata
                if not any(m["id"] == entry_id for m in self.metadata):
                    self.metadata.append(meta)
                    
            self.save_metadata()
            
        except Exception as e:
            print(f"Search failed: {e}")

    def save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata. Total papers: {len(self.metadata)}")

if __name__ == "__main__":
    collector = ArxivCollector()
    
    # Robotics
    collector.fetch_papers("robot navigation OR SLAM OR sensor fusion robotics", "robotics", 40)
    
    # RAG
    collector.fetch_papers("retrieval augmented generation OR self-rag", "rag", 20)
    
    # Hallucination
    collector.fetch_papers("llm hallucination OR uncertainty estimation llm", "hallucination", 20)
    
    # Scientific QA
    collector.fetch_papers("scientific question answering OR literature synthesis", "scientific_qa", 20)
    
    print("Collection phase complete!")
