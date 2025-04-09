from flask import Flask, render_template, request, send_file, jsonify
import spacy
import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
import tempfile
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import base64
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
from textblob import TextBlob
from gensim import corpora, models
from datetime import datetime
import re
from sklearn.tree import DecisionTreeClassifier, plot_tree
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from youtube_transcript_api import YouTubeTranscriptApi 
import pyLDAvis.gensim_models

app = Flask(__name__)

class MiniProject:
    plot_lock = threading.Lock()
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("Please install the en_core_web_sm model: python -m spacy download en_core_web_sm")
        self.model = None

    def initialize_model(self, api_key):
        if not api_key:
            raise ValueError("API key is required")
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {str(e)}")

    def process_chunks(self, chunks):
        if not chunks:
            raise ValueError("No text chunks provided")
        try:
            all_entities = set()
            for chunk in chunks:
                entities = self.extract_entities(chunk)
                all_entities.update(entities)
            
            # Track entities in relationships
            entities_in_rels = set()
            all_relationships = set()
            
            for chunk in chunks:
                relationships = self.generate_relationship(chunk, all_entities)
                for rel in relationships:
                    if '-[' in rel and ']->' in rel:
                        parts = rel.split(' -[')
                        entity_a = parts[0].strip()
                        rest = parts[1].split(']-> ')
                        if len(rest) >= 1:
                            entity_b = rest[1].strip()
                            entities_in_rels.add(entity_a)
                            entities_in_rels.add(entity_b)
                all_relationships.update(relationships)
            
            # Filter entities to those with relationships
            filtered_entities = list({
                (entity, label) for entity, label in all_entities 
                if entity in entities_in_rels
            })
            
            # Get labels from filtered entities
            labels = [label for _, label in filtered_entities]
            
            return filtered_entities, list(all_relationships), labels
        except Exception as e:
            raise Exception(f"Error processing chunks: {str(e)}")

    def read_pdf(self, file):
        if not file:
            raise ValueError("No file provided")
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
            return "\n".join(full_text)
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def read_docx(self, file):
        if not file:
            raise ValueError("No file provided")
        try:
            doc = docx.Document(file)
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n".join(full_text)
        except Exception as e:
            raise Exception(f"Error reading Word document: {str(e)}")

    def fetch_url_content(self, url):
        if not url:
            raise ValueError("No URL provided")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'meta', 'input']):
                tag.decompose()
            content_tags = soup.find_all(['p', 'article', 'section', 'div'], class_=lambda x: x and ('content' in x.lower() or 'article' in x.lower()))
            if not content_tags:
                content_tags = soup.find_all(['p', 'article', 'section'])
            text_content = ' '.join(p.get_text().strip() for p in content_tags if p.get_text().strip())
            if not text_content:
                raise ValueError("No content found on the page")
            return text_content
        except requests.RequestException as e:
            raise Exception(f"Error fetching URL content: {str(e)}")

    def extract_youtube_transcript(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([entry['text'] for entry in transcript])
            return text
        except Exception as e:
            raise Exception(f"Error extracting YouTube transcript: {str(e)}")

    def split_text_into_chunks(self, text, num_chunks=3):
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input")
        if num_chunks < 1:
            raise ValueError("Number of chunks must be positive")
        text = ' '.join(text.split())
        words = text.split()
        if not words:
            raise ValueError("Text contains no words")
        chunk_size = max(1, len(words) // num_chunks)
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size) if ' '.join(words[i:i + chunk_size]).strip()]
        while len(chunks) > num_chunks and len(chunks) > 1:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()
        return chunks

    def extract_entities(self, text):
        if not text or not isinstance(text, str):
            return []
        try:
            doc = self.nlp(text[:1000000])
            entities = set((ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ == "DATE" or not ent.text.replace('.', '').replace(',', '').isdigit())
            return list(entities)
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return []

    def generate_relationship(self, text, entities):
        if not text or not entities:
            return []
        try:
            entity_list = "\n".join([f"- {entity[0]} ({entity[1]})" for entity in entities])
            prompt = f"""
Analyze this text and create SPECIFIC relationships between entities. Use EXACT format:
Entity A -[specific relationship]-> Entity B
Entity C -[another relationship]-> Entity D
Text: {text[:2000]}
Entities:
{entity_list}
Generate 10-15 IMPORTANT relationships. Be precise and avoid generic connections.
Examples of GOOD relationships:
Microsoft -[developed]-> Windows
Einstein -[published theory of]-> Relativity
Amazon -[acquired]-> Twitch
BAD relationships to avoid:
Company -[has relationship with]-> Product
Person -[associated with]-> Concept
Now generate the relationships:
"""
            response = self.model.generate_content(prompt)
            raw_relationships = response.text.strip().split('\n')
            
            # Use set with canonical form to prevent duplicates
            valid_rels = set()
            pattern = r"(.+?)\s*-\[(.+?)\]-\>\s*(.+)"
            
            for rel in raw_relationships:
                match = re.match(pattern, rel)
                if match and len(match.groups()) == 3:
                    entity_a = match.group(1).strip()
                    relationship = match.group(2).strip()
                    entity_b = match.group(3).strip()
                    canonical = f"{entity_a}|{relationship}|{entity_b}"
                    valid_rels.add(canonical)
            
            # Convert back to relationship strings
            return [f"{rel.split('|')[0]} -[{rel.split('|')[1]}]-> {rel.split('|')[2]}" 
                    for rel in valid_rels][:7]
        
        except Exception as e:
            print(f"Relationship generation error: {str(e)}")
            return []

    def create_knowledge_graph(self, entities, relationships, title="Knowledge Graph"):
        """Enhanced static knowledge graph with relationship handling"""
        if not entities or not relationships:
            raise ValueError("No entities or relationships to visualize")
        try:
            G = nx.DiGraph()
            entities_in_rels = set()  # Track entities involved in relationships

            # Process relationships first to identify connected entities
            for rel in relationships:
                try:
                    if '-[' in rel and ']->' in rel:
                        parts = rel.split(' -[')
                        entity_a = parts[0].strip()
                        rest = parts[1].split(']-> ')
                        relationship = rest[0].strip()
                        entity_b = rest[1].strip()

                        # Add entities involved in relationships
                        entities_in_rels.add(entity_a)
                        entities_in_rels.add(entity_b)

                        # Add nodes and edges
                        if entity_a not in G:
                            G.add_node(entity_a, label='UNKNOWN')
                        if entity_b not in G:
                            G.add_node(entity_b, label='UNKNOWN')
                        G.add_edge(entity_a, entity_b, relationship=relationship)
                except Exception as e:
                    print(f"Error processing relationship: {rel} - {str(e)}")
                    continue

            # Filter entities to only those involved in relationships
            filtered_entities = [
                (entity, label) for entity, label in entities 
                if entity in entities_in_rels
            ]

            # Add remaining entities with proper labels
            for entity, label in filtered_entities:
                if entity not in G:
                    G.add_node(entity, label=label)
                else:
                    G.nodes[entity]['label'] = label

            # Ensure graph has valid nodes and edges
            if not G.nodes():
                raise ValueError("No valid nodes to visualize")

            # Define node colors based on entity type
            colors = {
                'PERSON': '#ADD8E6',  # Light Blue
                'ORG': '#90EE90',     # Light Green
                'GPE': '#FFB6C1',     # Light Pink
                'NORP': '#FFFACD',    # Pale Yellow
                'DATE': '#DDA0DD',    # Plum
                'LOC': '#98FB98',     # Pale Green
                'PRODUCT': '#F0E68C', # Khaki
                'EVENT': '#E6E6FA',   # Lavender
                'UNKNOWN': '#D3D3D3'  # Light Gray
            }

            # Draw the graph
            plt.figure(figsize=(8, 8))
            pos = nx.spring_layout(G, k=2.5, iterations=50)

            # Draw nodes with appropriate colors
            for node in G.nodes():
                node_type = G.nodes[node].get('label', 'UNKNOWN')
                color = colors.get(node_type, colors['UNKNOWN'])
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=[node],
                    node_color=color,
                    node_size=2000,
                    alpha=0.7
                )

            # Draw edges with styling
            nx.draw_networkx_edges(
                G, pos,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                width=1.5,
                alpha=0.6
            )

            # Draw node labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=8,
                font_weight='bold'
            )

            # Draw edge labels
            edge_labels = nx.get_edge_attributes(G, 'relationship')
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=6,
                font_color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )

            # Add title and finalize layout
            plt.title(title, pad=20, fontsize=16)
            plt.axis('off')
            plt.tight_layout()

            # Save the graph to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight', format='png')
                plt.close()
                return tmp_file.name

        except Exception as e:
            plt.close()
            raise Exception(f"Error creating knowledge graph: {str(e)}")

    def create_interactive_graph(self, entities, relationships):
        """Create interactive graph data structure for vis.js"""
        try:
            nodes = []
            edges = []
            node_set = set()
            entities_in_rels = set()

            # Process relationships
            for rel in relationships:
                try:
                    parts = re.split(r' -\[|\]->?', rel)
                    parts = [p.strip() for p in parts if p.strip()]
                    if len(parts) >= 3:
                        entity_a, relationship, entity_b = parts[0], parts[1], parts[-1]
                        
                        entities_in_rels.add(entity_a)
                        entities_in_rels.add(entity_b)

                        # Add nodes if not already present
                        for ent in [entity_a, entity_b]:
                            if ent not in node_set:
                                nodes.append({
                                    'id': ent,
                                    'label': ent,
                                    'group': 'AUTO',
                                    'shape': 'dot',
                                    'size': 20
                                })
                                node_set.add(ent)

                        # Add edge
                        edges.append({
                            'from': entity_a,
                            'to': entity_b,
                            'label': relationship,
                            'arrows': 'to'
                        })
                except Exception as e:
                    print(f"Skipping invalid relationship: {rel} - {str(e)}")

            # Add entities with labels
            for entity, label in entities:
                if entity in entities_in_rels and entity not in node_set:
                    nodes.append({
                        'id': entity,
                        'label': f"{entity} ({label})",
                        'group': label,
                        'shape': 'dot',
                        'size': 25
                    })
                    node_set.add(entity)

            return {
                'nodes': nodes,
                'edges': edges
            }
        except Exception as e:
            raise Exception(f"Interactive graph error: {str(e)}")

    def create_process_flow(self):
        """Create a simplified process flow visualization"""
        try:
            # Create figure
            plt.figure(figsize=(10, 4))
        
            # Define the process steps
            steps = ['Data Extraction', 'Text Cleaning', 'Entity Recognition', 'Relationship Generation']
        
            # Create x and y coordinates for steps
            x = np.linspace(0, 9, len(steps))  # Spread steps evenly
            y = np.zeros_like(x)  # All steps at same y-level
        
            # Plot points for each step
            plt.scatter(x, y, c='red', s=100, zorder=2)
        
            # Draw lines connecting steps
            plt.plot(x, y, 'gray', linestyle='-', linewidth=2, zorder=1)
         
            # Add step labels
            for i, (step, xi) in enumerate(zip(steps, x)):
                plt.annotate(
                    step,
                    xy=(xi, y[i]),
                    xytext=(0, 20),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        fc='white',
                        ec='gray',
                        alpha=0.8
                    )
                )
        
            # Configure plot
            plt.xlim(min(x) - 0.5, max(x) + 0.5)
            plt.ylim(-1, 2)
            plt.axis('off')
        
            # Save plot to base64 string
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            plt.close()
        
            # Encode and return
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            
        except Exception as e:
            plt.close()
            print(f"Error in create_process_flow: {str(e)}")
            return None

    def generate_wordcloud(self, text):
        """Generate wordcloud visualization"""
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white').generate(text)
        plt.figure(figsize=(8, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        return self._save_plot()

    def create_word_frequency_chart(self, text):
        """Create word frequency distribution chart"""
        words = text.lower().split()
        word_freq = Counter(words)
        
        plt.figure(figsize=(8, 6))
        freq_df = pd.DataFrame(list(word_freq.items()), 
                             columns=['Word', 'Frequency']).nlargest(20, 'Frequency')
        sns.barplot(data=freq_df, x='Word', y='Frequency')
        plt.xticks(rotation=45)
        return self._save_plot()

    def highlight_ner_entities(self, text):
        """Generate HTML with highlighted named entities"""
        doc = self.nlp(text)
        colors = {'PERSON': '#fca', 'ORG': '#afa', 'DATE': '#aaf', 'GPE': '#faa'}
        html = text
        for ent in reversed(doc.ents):
            color = colors.get(ent.label_, '#ddd')
            html = f"{html[:ent.start_char]}<span style='background-color:{color}'>{html[ent.start_char:ent.end_char]}</span>{html[ent.end_char:]}"
        return html

    def create_sentiment_analysis(self, text):
        """Generate sentiment analysis visualizations with improved error handling"""
        try:
            # Perform sentiment analysis
            blob = TextBlob(text)
            sentiments = [sentence.sentiment.polarity for sentence in blob.sentences]
        
            # Create figure with subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Sentiment Distribution', 'Sentiment Flow'),
                specs=[[{'type': 'pie'}, {'type': 'scatter'}]]
            )
        
            # Calculate sentiment categories
            categories = {
                'Positive': len([s for s in sentiments if s > 0]),
                'Neutral': len([s for s in sentiments if s == 0]),
                'Negative': len([s for s in sentiments if s < 0])
            }
        
            # Add pie chart
            fig.add_trace(
                go.Pie(
                    labels=list(categories.keys()),
                    values=list(categories.values()),
                    hole=0.3,
                    marker_colors=['#2ecc71', '#95a5a6', '#e74c3c']
                ),
                row=1, col=1
            )
        
            # Add sentiment flow line chart
            fig.add_trace(
                go.Scatter(
                    y=sentiments,
                    mode='lines+markers',
                    name='Sentiment Flow',
                    line=dict(color='#3498db'),
                    marker=dict(
                        color=[
                            '#2ecc71' if s > 0 else '#e74c3c' if s < 0 else '#95a5a6'
                            for s in sentiments
                        ]
                    )
                ),
                row=1, col=2
            )
        
            # Update layout
            fig.update_layout(
                title_text="Sentiment Analysis",
                showlegend=True,
                height=500,
                width=1000,
                template='plotly_white'
            )
        
            # Update axes for sentiment flow
            fig.update_yaxes(
                title_text="Sentiment Polarity",
                range=[-1.1, 1.1], 
                row=1, col=2
            )
            fig.update_xaxes(
                title_text="Sentence Number",
                row=1, col=2
            )
        
            return fig.to_html(full_html=False, include_plotlyjs=True)
        
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            # Return a simplified version if there's an error
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Positive', 'Neutral', 'Negative'],
                    values=[1, 1, 1],
                    hole=0.3,
                    marker_colors=['#2ecc71', '#95a5a6', '#e74c3c']
                )
            ])
            fig.update_layout(
                title_text="Sentiment Analysis (Error occurred - showing placeholder)",
                height=400,
                width=600
            )
            return fig.to_html(full_html=False, include_plotlyjs=True)

    def create_topic_modeling(self, text, num_topics=5):
        """Generate topic modeling visualizations with improved data handling"""
        try:
            # Tokenize and prepare text
            tokens = self.nlp(text)
            documents = [[token.text for token in tokens 
                     if not token.is_stop and token.is_alpha]]
         
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(documents)
            corpus = [dictionary.doc2bow(doc) for doc in documents]
        
            # Train LDA model
            lda_model = models.LdaModel(corpus, num_topics=num_topics, 
                                  id2word=dictionary)
        
            # Create visualization data
            topics_dict = {}
            for idx, topic in lda_model.show_topics(formatted=False):
                words, scores = zip(*topic)
                topics_dict[f"Topic {idx+1}"] = {
                    'words': words[:5],  # Get top 5 words
                    'scores': [float(score) for score in scores[:5]]  # Convert scores to float
                }
        
            # Create figure
            fig = go.Figure()
        
            # Add a trace for each topic
            for topic_idx, (topic_name, topic_data) in enumerate(topics_dict.items()):
                fig.add_trace(go.Bar(
                    name=topic_name,
                    x=list(topic_data['words']),  # Convert words to list
                    y=topic_data['scores'],  # Use probability scores
                    text=topic_data['words'],
                    textposition='auto',
                ))

            # Update layout
            fig.update_layout(
                title="Topic Modeling Results",
                xaxis_title="Top Words",
                yaxis_title="Word Probability Score",
                barmode='group',
                height=400,
                width=800,
                showlegend=True,
                template='plotly_white'
            )
        
            return fig.to_html(full_html=False, include_plotlyjs=True)
        
        except Exception as e:
            print(f"Error in topic modeling: {str(e)}")
            # Return a simplified version if there's an error
            fig = go.Figure(data=[
                go.Bar(
                    x=['Error occurred', 'Please try', 'with different', 'text', 'input'],
                    y=[1, 1, 1, 1, 1],
                    text=['Error occurred', 'Please try', 'with different', 'text', 'input'],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Topic Modeling (Error occurred - showing placeholder)",
                height=400,
                width=600
            )
            return fig.to_html(full_html=False, include_plotlyjs=True)

    def create_timeline(self, text):
        """Generate timeline visualization from dates in text"""
        doc = self.nlp(text)
        dates = []
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                try:
                    date = pd.to_datetime(ent.text, dayfirst=True)
                    dates.append({'date': date, 'event': ent.sent.text})
                except:
                    continue
        
        if dates:
            df = pd.DataFrame(dates)
            fig = px.timeline(df, x_start='date', x_end='date', text='event')
            return fig.to_html()
        return None

    def create_decision_tree(self, text, labels):
        """Generate decision tree visualization with proper handling of text chunks and labels"""
        try:
            # Convert text to list if string provided
            if isinstance(text, str):
                text = [text]
            
            # Create vectorizer and transform text
            vectorizer = TfidfVectorizer(max_features=10)
            X = vectorizer.fit_transform(text)
        
            # Handle labels
            if labels is None or len(labels) != X.shape[0]:
                # Use sentiment analysis to create binary labels matching the number of samples
                from textblob import TextBlob
                labels = ['positive' if TextBlob(t).sentiment.polarity > 0 else 'negative' 
                        for t in text]
            else:
                # If labels provided, ensure we have the right number
                # Take the most frequent labels to match number of samples
                from collections import Counter
                label_counts = Counter(labels)
                most_common_labels = [label for label, _ in label_counts.most_common()][:X.shape[0]]
                # Pad with the most common label if needed
                while len(most_common_labels) < X.shape[0]:
                    most_common_labels.append(label_counts.most_common(1)[0][0])
                labels = most_common_labels[:X.shape[0]]
        
            # Ensure we have enough samples for each class
            unique_labels = list(set(labels))
            if len(unique_labels) < 2:
                # If not enough unique labels, default to binary sentiment
                labels = ['positive' if i % 2 == 0 else 'negative' for i in range(X.shape[0])]
                unique_labels = ['positive', 'negative']
            
            # Create and fit decision tree
            clf = DecisionTreeClassifier(max_depth=3)
            clf.fit(X, labels)
        
            # Generate visualization
            plt.figure(figsize=(10, 6))
            plot_tree(clf, 
                 feature_names=vectorizer.get_feature_names_out(),
                 class_names=unique_labels,
                 filled=True,
                 rounded=True,
                 fontsize=8)
        
            return self._save_plot()
        
        except Exception as e:
            print(f"Error creating decision tree: {str(e)}")
            # Return a simple visualization indicating the error
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, f'Unable to create decision tree:\n{str(e)}',
                ha='center', va='center', wrap=True)
            plt.axis('off')
            return self._save_plot()

    def create_interactive_table(self, data):
        """Create interactive data table visualization"""
        fig = go.Figure(data=[go.Table(
               header=dict(values=list(data.columns),
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[data[col] for col in data.columns],
                      fill_color='lavender',
                      align='left'))
        ])
        return fig.to_html()

    def _save_plot(self):
        """Save plot to base64 string with proper cleanup"""
        try:
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            plt.close('all')
            # Add data URL prefix
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        finally:
            buf.close()

    def generate_explanation(self, entities, relationships): 
        """Generate explanation for the knowledge graph using Gemini"""
        if not entities or not relationships: 
            return "Insufficient data to generate explanation."

        try:
            # Format input data
            entity_list = "\n".join([f"- {entity[0]} ({entity[1]})" for entity in entities])
            relationship_list = "\n".join(relationships)
            
            prompt = f"""
Analyze these entities and relationships from a knowledge graph and provide a clear, structured explanation 
of the main insights and patterns discovered. Focus on practical implications and meaningful connections.

Entities:
{entity_list}
 
Relationships:
{relationship_list}

Please provide:
1. Overview: Brief summary of the main entities and their roles (2-3 sentences)
2. Key Relationships: Most significant connections and their implications (2-3 points)
3. Insights: Important patterns or conclusions that can be drawn (2-3 points)
4. Recommendations: Suggested actions or areas for further investigation (1-2 points)

Keep the analysis concise, specific, and focused on actionable insights.
"""
            # Add retry mechanism for API calls 
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    explanation = response.text.strip()
                    
                    # Validate explanation
                    if len(explanation) < 50:
                        raise ValueError("Generated explanation is too short")
                        
                    return explanation
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    continue
                    
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

    def generate_summary(self, text):
        """Generate a point-wise summary of the main content using Gemini"""
        if not text or not isinstance(text, str):
            return "No content to summarize."

        try:
            prompt = f"""
Analyze the following text and create a structured, point-wise summary of the main content.
Focus on key topics, arguments, and findings.

Text: {text[:3000]}  # Limit text length for API

Please provide:
a breif comprehensive summary of the provided content in paragraph form. Focus on capturing the main themes, significant arguments, and key conclusions in a cohesive and concise manner. Ensure the summary flows logically and includes only essential information, presenting a clear understanding of the overall topic without breaking it down into sections

Format each point with a bullet point (•) and keep points concise and clear.
"""
            # Add retry mechanism for API calls
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    summary = response.text.strip()
                
                    # Validate summary
                    if len(summary) < 50:
                        raise ValueError("Generated summary is too short")
                    
                    return summary
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    continue
                
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def create_pdf_report(self, entities, relationships, graph_file, explanation, text_content="", 
                            author_name="Thirumani Srikanth", 
                            contact_number="7671813675", 
                            email="srikanththirumani01@gmail.com"):
        """Create a professional PDF report with auto-adjusting tables and proper content wrapping"""
        import textwrap
        from reportlab.platypus import KeepTogether
        
        if not all([entities, relationships, explanation]):
            raise ValueError("Missing required components for PDF report")
        
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=0.5*inch, rightMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
            styles = getSampleStyleSheet()
            story = []
            
            # Define custom styles with professional color scheme
            primary_color = colors.HexColor('#2C3E50')
            accent_color = colors.HexColor('#3498DB')
            light_bg = colors.HexColor('#ECF0F1')
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName='Helvetica-Bold',
                fontSize=24,
                spaceAfter=15,
                alignment=1,
                textColor=primary_color
            )
            
            section_style = ParagraphStyle(
                'SectionStyle',
                parent=styles['Heading2'],
                fontName='Helvetica-Bold',
                fontSize=16,
                spaceBefore=20,
                spaceAfter=10,
                textColor=accent_color
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=10,
                spaceBefore=6,
                spaceAfter=6,
                leading=14
            )
            
            table_text_style = ParagraphStyle(
                'TableText',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=9,
                leading=11,
                spaceBefore=3,
                spaceAfter=3
            )
            
            author_style = ParagraphStyle(
                'AuthorStyle',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=10,
                alignment=1,
                textColor=primary_color
            )

            # Header section
            header_text = f"""
            <para align="center" fontSize="8" textColor="#7f8c8d">
            Professional Knowledge Graph Analysis • {datetime.now().strftime('%B %d, %Y')}
            </para>
            """
            story.append(Paragraph(header_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Title
            story.append(Paragraph("Knowledge Graph Analysis Report", title_style))
            
            # Author information
            contact_info = f"""
            <para align="center" fontSize="11">
            Prepared by: <b>{author_name}</b><br/>
            Contact: {contact_number} | {email}
            </para>
            """
            story.append(Paragraph(contact_info, author_style))
            story.append(Spacer(1, 30))
            
            # Content summary
            if text_content:
                summary = self.generate_summary(text_content)
                story.append(Paragraph("Content Summary", section_style))
                for section in summary.split('\n'):
                    if section.strip():
                        story.append(Paragraph(section.strip(), normal_style))
                story.append(Spacer(1, 20))

            # ENTITIES TABLE WITH AUTO-SIZING
            if entities:
                story.append(Paragraph("Identified Entities", section_style))
                
                # Calculate maximum content lengths
                max_entity_len = max(len(entity[0]) for entity in entities) if entities else 20
                max_type_len = max(len(entity[1]) for entity in entities) if entities else 15
                
                # Set column widths (with min/max constraints)
                entity_col_width = min(max(max_entity_len * 0.07, 1.5), 3.5) * inch
                type_col_width = min(max(max_type_len * 0.07, 1.0), 2.5) * inch
                
                # Prepare table data with wrapped text
                entity_data = []
                for entity in entities:
                    # Wrap text and convert to Paragraphs for proper rendering
                    wrapped_entity = Paragraph(
                        '<br/>'.join(textwrap.wrap(entity[0], width=25)),
                        table_text_style
                    )
                    wrapped_type = Paragraph(
                        '<br/>'.join(textwrap.wrap(entity[1], width=20)),
                        table_text_style
                    )
                    entity_data.append([wrapped_entity, wrapped_type])
                
                # Add header row
                header_row = [
                    Paragraph('<b>Entity</b>', table_text_style),
                    Paragraph('<b>Type</b>', table_text_style)
                ]
                entity_data.insert(0, header_row)
                
                # Create table
                entity_table = Table(
                    entity_data,
                    colWidths=[entity_col_width, type_col_width],
                    repeatRows=1
                )
                
                # Style table
                entity_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('TOPPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BOX', (0, 0), (-1, -1), 1, primary_color),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [light_bg, colors.white]),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
                    ('TOPPADDING', (0, 1), (-1, -1), 5),
                ]))
                
                story.append(KeepTogether(entity_table))
                story.append(Spacer(1, 20))

            # RELATIONSHIPS TABLE WITH AUTO-SIZING
            if relationships:
                story.append(Paragraph("Entity Relationships", section_style))
                
                relationship_data = []
                for idx, rel in enumerate(relationships, 1):
                    if '-[' in rel and ']->' in rel:
                        # Parse relationship components
                        parts = rel.split(' -[')
                        entity_a = parts[0].strip()
                        rest = parts[1].split(']-> ')
                        relationship = rest[0].strip()
                        entity_b = rest[1].strip()
                        
                        # Create wrapped Paragraph objects
                        wrapped_entity_a = Paragraph(
                            '<br/>'.join(textwrap.wrap(entity_a, width=25)),
                            table_text_style
                        )
                        wrapped_rel = Paragraph(
                            '<br/>'.join(textwrap.wrap(relationship, width=20)),
                            table_text_style
                        )
                        wrapped_entity_b = Paragraph(
                            '<br/>'.join(textwrap.wrap(entity_b, width=25)),
                            table_text_style
                        )
                        
                        relationship_data.append([
                            Paragraph(str(idx) + ".", table_text_style),
                            wrapped_entity_a,
                            wrapped_rel,
                            wrapped_entity_b
                        ])
                    else:
                        # Fallback for malformed relationships
                        wrapped_content = Paragraph(
                            '<br/>'.join(textwrap.wrap(rel, width=60)),
                            table_text_style
                        )
                        relationship_data.append([
                            Paragraph(str(idx) + ".", table_text_style),
                            wrapped_content,
                            Paragraph("", table_text_style),
                            Paragraph("", table_text_style)
                        ])
                
                if relationship_data:
                    # Calculate column widths based on content
                    num_cols = 4
                    col_widths = [
                        0.5 * inch,  # Index column
                        2.5 * inch,  # Entity A
                        2.0 * inch,  # Relationship
                        2.5 * inch   # Entity B
                    ]
                    
                    # Add header row
                    header_row = [
                        Paragraph('<b>#</b>', table_text_style),
                        Paragraph('<b>Entity A</b>', table_text_style),
                        Paragraph('<b>Relationship</b>', table_text_style),
                        Paragraph('<b>Entity B</b>', table_text_style)
                    ]
                    relationship_data.insert(0, header_row)
                    
                    # Create table
                    rel_table = Table(
                        relationship_data,
                        colWidths=col_widths,
                        repeatRows=1
                    )
                    
                    # Style table
                    rel_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                        ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('TOPPADDING', (0, 0), (-1, 0), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('BOX', (0, 0), (-1, -1), 1, primary_color),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [light_bg, colors.white]),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
                        ('TOPPADDING', (0, 1), (-1, -1), 5),
                    ]))
                    
                    story.append(KeepTogether(rel_table))
                    story.append(Spacer(1, 20))

            # KNOWLEDGE GRAPH VISUALIZATION
            if graph_file and os.path.exists(graph_file):
                story.append(Paragraph("Knowledge Graph Visualization", section_style))
                img = Image(graph_file)
                
                # Set image dimensions (maintain aspect ratio)
                img_width = 6.5 * inch
                img_height = 6.5 * inch * (img.drawHeight / img.drawWidth)
                img.drawWidth = img_width
                img.drawHeight = img_height
                
                # Add frame around image
                img_frame = Table([[img]], colWidths=[img_width])
                img_frame.setStyle(TableStyle([
                    ('BOX', (0, 0), (-1, -1), 1, accent_color),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                
                story.append(Spacer(1, 10))
                story.append(img_frame)
                story.append(Spacer(1, 20))

            # ANALYSIS AND INSIGHTS
            if explanation:
                story.append(Paragraph("Analysis and Insights", section_style))
                
                for paragraph in explanation.split('\n'):
                    if paragraph.strip():
                        if paragraph.strip().startswith('**') and ':**' in paragraph:
                            # Format section headers
                            header_text = paragraph.split(':**')[0].replace('**', '')
                            content_text = paragraph.split(':**')[1]
                            story.append(Paragraph(
                                f"<b>{header_text}:</b> {content_text}",
                                normal_style
                            ))
                        else:
                            story.append(Paragraph(paragraph.strip(), normal_style))
                        
                        story.append(Spacer(1, 6))

            # FOOTER WITH PAGE NUMBERS
            def add_page_number(canvas, doc):
                canvas.saveState()
                canvas.setFont('Helvetica', 8)
                canvas.setFillColor(colors.grey)
                page_num = canvas.getPageNumber()
                page_num_text = f"Page {page_num}"
                canvas.drawRightString(7.5*inch, 0.5*inch, page_num_text)
                canvas.restoreState()
            
            # BUILD DOCUMENT
            doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
            return buffer
        
        except Exception as e:
            raise Exception(f"Error creating professional PDF report: {str(e)}")

project = MiniProject()

@app.route('/')
def index():
    try:
        return render_template('landing.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/index.html')
def index_page():
    try:
        return render_template('index.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process():
    try:
        # Step 1: Validate API Key
        api_key = request.form.get('api_key')
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        # Initialize the model with the provided API key
        project.initialize_model(api_key)

        # Step 2: Validate Input Method
        input_method = request.form.get('input_method')
        if not input_method:
            return jsonify({'error': 'Input method not specified'}), 400

        # Extract text based on the selected input method
        text = None
        if input_method == 'text':
            text = request.form.get('text')
            if not text:
                return jsonify({'error': 'No text provided'}), 400
        elif input_method in ['pdf', 'docx']:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            file = request.files['file']
            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400
            text = project.read_pdf(file) if input_method == 'pdf' else project.read_docx(file)
        elif input_method == 'url':
            url = request.form.get('url')
            if not url:
                return jsonify({'error': 'No URL provided'}), 400
            text = project.fetch_url_content(url)
        elif input_method == 'youtube':
            youtube_url = request.form.get('youtube_url')
            if not youtube_url:
                return jsonify({'error': 'No YouTube URL provided'}), 400
            video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
            if not video_id_match:
                return jsonify({'error': 'Invalid YouTube URL'}), 400
            video_id = video_id_match.group(1)
            text = project.extract_youtube_transcript(video_id)

        # Ensure text was successfully extracted
        if not text:
            return jsonify({'error': 'Failed to extract content'}), 400

        # Step 3: Process Text into Chunks
        chunks = project.split_text_into_chunks(text, 3)
        entities, relationships, labels = project.process_chunks(chunks)

        # Step 4: Filter Entities Based on Relationships
        entities_in_rels = set()
        for rel in relationships:
            if '-[' in rel and ']->' in rel:
                parts = rel.split(' -[')
                entity_a = parts[0].strip()
                rest = parts[1].split(']-> ')
                relationship = rest[0].strip()
                entity_b = rest[1].strip()
                entities_in_rels.add(entity_a)
                entities_in_rels.add(entity_b)

        # Filter entities to only those involved in relationships
        filtered_entities = [
            (entity, label) for entity, label in entities 
            if entity in entities_in_rels
        ]

        # Ensure meaningful relationships were found
        if not relationships or not filtered_entities:
            return jsonify({'error': 'No meaningful relationships found in the content'}), 400

        # Step 5: Generate Both Static and Interactive Graphs
        # Generate static knowledge graph
        static_graph_file = project.create_knowledge_graph(filtered_entities, relationships)
        with open(static_graph_file, 'rb') as f:
            static_graph_base64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()

        # Generate interactive knowledge graph
        interactive_graph_data = project.create_interactive_graph(filtered_entities, relationships)

        # Step 6: Generate Explanation and Summary
        explanation = project.generate_explanation(filtered_entities, relationships)
        summary = project.generate_summary(text)

        # Step 7: Generate Visualizations
        visualizations = {
            'process_flow': project.create_process_flow(),
            'wordcloud': project.generate_wordcloud(text),
            'word_frequency': project.create_word_frequency_chart(text),
            'ner_text': project.highlight_ner_entities(text[:1000]),
            'sentiment': project.create_sentiment_analysis(text),
            'topics': project.create_topic_modeling(text),
        }

        # Optional: Add decision tree visualization if possible
        try:
            decision_tree = project.create_decision_tree(chunks, labels)
            if decision_tree:
                visualizations['decision'] = decision_tree
        except Exception as e:
            print(f"Warning: Could not create decision tree: {e}")

        # Optional: Add timeline visualization if dates are present
        timeline = project.create_timeline(text)
        if timeline:
            visualizations['timeline'] = timeline

        # Create interactive table for entities
        entity_data = pd.DataFrame(filtered_entities, columns=['Entity', 'Type'])
        visualizations['entity_table'] = project.create_interactive_table(entity_data)

        # Step 8: Generate PDF Report
        pdf_buffer = project.create_pdf_report(
            filtered_entities, relationships, static_graph_file, explanation, text
        )

        # Clean up temporary graph file if it exists
        try:
            if static_graph_file and os.path.exists(static_graph_file):
                os.unlink(static_graph_file)
        except Exception as e:
            print(f"Error removing temporary file: {str(e)}")

        # Step 9: Format Relationships for Response
        relationship_data = []
        for rel in relationships:
            if '-[' in rel and ']->' in rel:
                parts = rel.split(' -[')
                entity_a = parts[0].strip()
                rest = parts[1].split(']-> ')
                relationship = rest[0].strip()
                entity_b = rest[1].strip()
                relationship_data.append({
                    'entity_a': entity_a,
                    'relationship': relationship,
                    'entity_b': entity_b
                })

        # Step 10: Return JSON Response with Both Graphs
        return jsonify({
            'success': True,
            'relationships': relationship_data,
            'static_graph': static_graph_base64,
            'interactive_graph_data': interactive_graph_data,
            'explanation': explanation,
            'pdf': base64.b64encode(pdf_buffer.getvalue()).decode(),
            'visualizations': visualizations,
            'summary': summary,
            'text_stats': {
                'total_entities': len(filtered_entities),
                'total_relationships': len(relationships),
                'text_length': len(text),
                'chunk_count': len(chunks)
            }
        })
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({
            'error': str(e),
            'details': f"An error occurred while processing your request: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
