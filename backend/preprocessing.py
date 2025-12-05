import json
import os
import re
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from config import DATA_DIR, PROCESSED_DIR

class DataPreprocessor:
    def __init__(self):
        self.raw_prs_path = os.path.join(DATA_DIR, "prs.json")
        self.raw_issues_path = os.path.join(DATA_DIR, "issues.json")
        
        # Load Raw Data
        with open(self.raw_prs_path, 'r', encoding='utf-8') as f:
            self.prs = json.load(f)
        with open(self.raw_issues_path, 'r', encoding='utf-8') as f:
            self.issues = json.load(f)
            
        print(f"📊 Loaded: {len(self.prs)} PRs, {len(self.issues)} Issues")

    def clean_text(self, text):
        """Remove markdown, code blocks, and noise"""
        if not isinstance(text, str): 
            return ""
        
        # Remove code blocks
        text = re.sub(r'``````', '', text)
        text = re.sub(r'`[^`]*`', '', text)
        # Remove links
        text = re.sub(r'http\S+', '', text)
        # Remove special markdown
        text = re.sub(r'[*_#\[\](){}<>]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        return text

    def extract_issue_links(self, pr_body):
        """Find explicit issue references"""
        if not pr_body: 
            return []
        pattern = r'(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+#(\d+)'
        matches = re.findall(pattern, pr_body, re.IGNORECASE)
        return [int(m) for m in matches]

    def text_similarity(self, text1, text2):
        """Simple word overlap similarity"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        overlap = len(words1 & words2)
        max_len = max(len(words1), len(words2))
        return overlap / max_len if max_len > 0 else 0.0

    def process_data(self):
        print("\n🧹 Cleaning Data & Building Labels...")
        
        # 1. Convert Issues to DataFrame
        issues_df = []
        for issue in self.issues:
            issues_df.append({
                'issue_id': issue['id'],
                'repo': issue['repo'],
                'title': self.clean_text(issue['title']),
                'body': self.clean_text(issue['body'] or ''),
                'text': self.clean_text(f"{issue['title']} {issue['body'] or ''}")
            })
        self.df_issues = pd.DataFrame(issues_df)
        print(f"✅ Issues: {len(self.df_issues)}")
        
        # 2. Process PRs
        prs_cleaned = []
        
        for pr in tqdm(self.prs, desc="Processing PRs"):
            cleaned_body = self.clean_text(pr['body'] or '')
            cleaned_title = self.clean_text(pr['title'])
            
            prs_cleaned.append({
                'pr_id': pr['id'],
                'repo': pr['repo'],
                'title': cleaned_title,
                'body': cleaned_body,
                'text': f"{cleaned_title} {cleaned_body}",
                'files': " ".join(pr['files'][:5]) if pr['files'] else "",  # Top 5 files
                'user': pr['user'],
                'explicit_links': self.extract_issue_links(pr['body'])
            })
        
        self.df_prs = pd.DataFrame(prs_cleaned)
        print(f"✅ PRs: {len(self.df_prs)}")
        
        # 3. Create Training Labels
        print("\n📝 Creating Training Labels...")
        labels = []
        
        for pr_idx, pr in enumerate(tqdm(self.df_prs.itertuples(), total=len(self.df_prs), desc="Creating labels")):
            
            # Get top 10 most similar issues for this PR
            similarities = []
            for issue_idx, issue in self.df_issues.iterrows():
                sim = self.text_similarity(pr.text, issue['text'])
                similarities.append((issue['issue_id'], sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Label top 3 as positive (similar to PR), rest as negative
            for rank, (issue_id, sim) in enumerate(similarities[:10]):
                # Positive: top 3 matches (label=1)
                # Negative: rest (label=0)
                label = 1 if rank < 3 and sim > 0.1 else 0
                
                labels.append({
                    'pr_id': pr.pr_id,
                    'issue_id': issue_id,
                    'pr_title': pr.title[:80],
                    'issue_title': self.df_issues.loc[self.df_issues['issue_id'] == issue_id, 'title'].values[0][:80],
                    'similarity': sim,
                    'label': label
                })
        
        self.df_labels = pd.DataFrame(labels)
        
        positive_count = (self.df_labels['label'] == 1).sum()
        negative_count = (self.df_labels['label'] == 0).sum()
        print(f"✅ Created {len(labels)} label pairs")
        print(f"   Positive (1): {positive_count}")
        print(f"   Negative (0): {negative_count}")

    def build_developer_graph(self):
        print("\n🕸️ Building Developer Graph...")
        G = nx.Graph()
        
        # Add developers as nodes
        for _, pr in self.df_prs.iterrows():
            user = pr['user']
            repo = pr['repo']
            if user and repo:
                G.add_node(user, type='developer')
                if G.has_edge(user, repo):
                    G[user][repo]['weight'] += 1
                else:
                    G.add_edge(user, repo, weight=1, type='contributes')
        
        self.G = G
        print(f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    def save_data(self):
        print("\n💾 Saving Processed Data...")
        self.df_prs.to_csv(os.path.join(PROCESSED_DIR, "pr_clean.csv"), index=False)
        self.df_issues.to_csv(os.path.join(PROCESSED_DIR, "issue_clean.csv"), index=False)
        self.df_labels.to_csv(os.path.join(PROCESSED_DIR, "labels.csv"), index=False)
        
        # Save Graph
        nx.write_gexf(self.G, os.path.join(PROCESSED_DIR, "dev_graph.gexf"))
        
        print("✅ All files saved:")
        print(f"   - pr_clean.csv ({len(self.df_prs)} rows)")
        print(f"   - issue_clean.csv ({len(self.df_issues)} rows)")
        print(f"   - labels.csv ({len(self.df_labels)} rows)")
        print(f"   - dev_graph.gexf")

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.process_data()
    processor.build_developer_graph()
    processor.save_data()
