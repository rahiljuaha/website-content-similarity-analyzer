import streamlit as st
import re
import string
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation,
    and basic cleaning while preserving word boundaries for keyword analysis.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep spaces between words
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text.strip()

def extract_keywords(text, vectorizer, feature_names, top_n=10):
    """
    Extract top keywords from text based on TF-IDF scores.
    """
    if not text.strip():
        return []
    
    # Transform the text using the fitted vectorizer
    tfidf_vector = vectorizer.transform([text])
    
    # Get feature scores
    feature_scores = tfidf_vector.toarray()[0]
    
    # Create keyword-score pairs
    keyword_scores = [(feature_names[i], feature_scores[i]) for i in range(len(feature_scores)) if feature_scores[i] > 0]
    
    # Sort by score and return top N
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_scores[:top_n]

def find_keyword_overlap(own_keywords, competitor_keywords):
    """
    Find overlapping keywords between own content and competitor content.
    """
    own_words = set([kw[0] for kw in own_keywords])
    comp_words = set([kw[0] for kw in competitor_keywords])
    
    overlap = own_words.intersection(comp_words)
    
    # Get scores for overlapping words
    own_dict = dict(own_keywords)
    comp_dict = dict(competitor_keywords)
    
    overlap_details = []
    for word in overlap:
        overlap_details.append({
            'keyword': word,
            'own_score': own_dict.get(word, 0),
            'competitor_score': comp_dict.get(word, 0)
        })
    
    # Sort by combined importance
    overlap_details.sort(key=lambda x: x['own_score'] + x['competitor_score'], reverse=True)
    
    return overlap_details

def analyze_seed_keywords(texts, vectorizer, feature_names, seed_keywords_list):
    """
    Analyze how seed keywords appear across all texts.
    """
    if not seed_keywords_list:
        return None
    
    # Clean and prepare seed keywords
    cleaned_seeds = [preprocess_text(kw.strip()) for kw in seed_keywords_list if kw.strip()]
    
    if not cleaned_seeds:
        return None
    
    # Get TF-IDF matrix for all texts
    tfidf_matrix = vectorizer.transform(texts)
    
    # Find seed keywords in feature names
    seed_analysis = []
    for seed in cleaned_seeds:
        if seed in feature_names:
            # Get index of the seed keyword
            feature_idx = list(feature_names).index(seed)
            
            # Get scores for this keyword across all texts
            scores = [tfidf_matrix[i, feature_idx] for i in range(len(texts))]
            
            seed_analysis.append({
                'keyword': seed,
                'your_score': scores[-1],  # Own content is last
                'comp1_score': scores[0],
                'comp2_score': scores[1],
                'comp3_score': scores[2],
                'total_score': sum(scores)
            })
    
    # Sort by total importance
    seed_analysis.sort(key=lambda x: x['total_score'], reverse=True)
    
    return seed_analysis

def create_similarity_chart(similarity_scores, competitor_names):
    """
    Create an interactive bar chart showing similarity scores.
    """
    fig = go.Figure(data=[
        go.Bar(
            x=competitor_names,
            y=similarity_scores,
            text=[f'{score:.1%}' for score in similarity_scores],
            textposition='auto',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
    ])
    
    fig.update_layout(
        title='Content Similarity Scores',
        xaxis_title='Competitors',
        yaxis_title='Similarity Score',
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        showlegend=False,
        height=400
    )
    
    return fig

def create_keyword_radar_chart(seed_analysis):
    """
    Create a radar chart showing seed keyword performance.
    """
    if not seed_analysis or len(seed_analysis) < 2:
        return None
    
    keywords = [item['keyword'] for item in seed_analysis[:5]]  # Top 5 keywords
    your_scores = [item['your_score'] for item in seed_analysis[:5]]
    comp1_scores = [item['comp1_score'] for item in seed_analysis[:5]]
    comp2_scores = [item['comp2_score'] for item in seed_analysis[:5]]
    comp3_scores = [item['comp3_score'] for item in seed_analysis[:5]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=your_scores,
        theta=keywords,
        fill='toself',
        name='Your Content',
        line_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=comp1_scores,
        theta=keywords,
        fill='toself',
        name='Competitor 1',
        line_color='#4ECDC4'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=comp2_scores,
        theta=keywords,
        fill='toself',
        name='Competitor 2',
        line_color='#45B7D1'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=comp3_scores,
        theta=keywords,
        fill='toself',
        name='Competitor 3',
        line_color='#96CEB4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(your_scores), max(comp1_scores), max(comp2_scores), max(comp3_scores))]
            )),
        showlegend=True,
        title="Seed Keywords Performance Comparison",
        height=500
    )
    
    return fig

def generate_comprehensive_report(similarity_scores, competitor_names, own_keywords, seed_analysis, overlap_data):
    """
    Generate a comprehensive analysis report.
    """
    report = f"""
# Website Content Similarity Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report analyzes the content similarity between your website and three competitors using advanced TF-IDF vectorization and cosine similarity metrics.

## Similarity Scores Overview
"""
    
    for i, (name, score) in enumerate(zip(competitor_names, similarity_scores)):
        report += f"- **{name}:** {score:.1%} similarity\n"
    
    avg_similarity = similarity_scores.mean()
    report += f"\n**Average Similarity:** {avg_similarity:.1%}\n"
    
    # Interpretation
    if avg_similarity > 0.6:
        interpretation = "HIGH - Your content is very similar to competitors. Consider developing more unique value propositions."
    elif avg_similarity > 0.3:
        interpretation = "MODERATE - Good balance between industry relevance and unique positioning."
    else:
        interpretation = "LOW - Your content is quite unique, which can be an advantage for differentiation."
    
    report += f"**Interpretation:** {interpretation}\n"
    
    # Top Keywords
    if own_keywords:
        report += f"\n## Your Top Keywords\n"
        for i, (keyword, score) in enumerate(own_keywords[:10], 1):
            report += f"{i}. {keyword} (Score: {score:.3f})\n"
    
    # Seed Keywords Analysis
    if seed_analysis:
        report += f"\n## Seed Keywords Performance\n"
        for item in seed_analysis:
            report += f"- **{item['keyword']}:** Your Score: {item['your_score']:.3f}, "
            report += f"Best Competitor: {max(item['comp1_score'], item['comp2_score'], item['comp3_score']):.3f}\n"
    
    # Recommendations
    report += f"\n## Recommendations\n"
    
    max_sim_idx = similarity_scores.argmax()
    min_sim_idx = similarity_scores.argmin()
    
    report += f"1. **Focus on differentiation from {competitor_names[max_sim_idx]}** - They have the highest similarity ({similarity_scores[max_sim_idx]:.1%})\n"
    report += f"2. **Learn from {competitor_names[min_sim_idx]}** - They use different messaging strategies ({similarity_scores[min_sim_idx]:.1%} similarity)\n"
    
    if avg_similarity > 0.5:
        report += "3. **Develop unique content** - Your overall similarity is high, focus on creating distinctive messaging\n"
    
    if seed_analysis:
        underperforming = [item for item in seed_analysis if item['your_score'] < 0.05]
        if underperforming:
            report += f"4. **Strengthen these keywords:** {', '.join([item['keyword'] for item in underperforming[:3]])}\n"
    
    report += f"\n## Methodology\n"
    report += "- **Text Processing:** Lowercase conversion, punctuation removal, stop word filtering\n"
    report += "- **Vectorization:** TF-IDF with unigrams and bigrams\n"
    report += "- **Similarity Metric:** Cosine similarity\n"
    report += "- **Keyword Extraction:** TF-IDF score ranking\n"
    
    return report

# Streamlit App Configuration
st.set_page_config(
    page_title="Website Content Similarity Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    # Advanced settings
    st.subheader("TF-IDF Configuration")
    max_features = st.slider("Maximum Features", 500, 5000, 1000, 250, help="Maximum number of features to extract")
    min_df = st.slider("Minimum Document Frequency", 1, 5, 1, help="Ignore terms that appear in fewer than this many documents")
    max_df = st.slider("Maximum Document Frequency", 0.8, 1.0, 0.95, 0.05, help="Ignore terms that appear in more than this fraction of documents")
    
    include_bigrams = st.checkbox("Include Bigrams", value=True, help="Include two-word phrases in analysis")
    ngram_range = (1, 2) if include_bigrams else (1, 1)
    
    st.subheader("Display Options")
    top_keywords_count = st.slider("Top Keywords to Show", 5, 25, 15, 5)
    
    # Export options
    st.subheader("📁 Quick Actions")
    if st.button("🎯 Focus Mode", help="Hide sidebar for better viewing"):
        st.session_state.sidebar_state = "collapsed"
    
    # Analysis tips
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Higher similarity** means more overlapping content
    - **Seed keywords** help track specific terms
    - **Sample data** shows how the tool works
    - **Download reports** for sharing results
    """)

# Main Title and Description
st.title("📊 Website Content Similarity Analyzer")
st.markdown("""
This advanced tool analyzes how similar your website content is to your competitors using machine learning and natural language processing.
Get detailed similarity scores, keyword overlap analysis, interactive visualizations, and downloadable reports.
""")

# Instructions and How-to Guide
with st.expander("📚 How to Use This Tool & Gain Insights", expanded=False):
    st.markdown("""
    ## 🚀 Quick Start Guide
    
    ### Step 1: Input Your Content
    1. **Use Sample Data**: Click "Load Sample Data" to see how the analysis works with example content
    2. **Add Your Content**: Paste text from your website and three competitors into the text boxes
    3. **Add Seed Keywords**: Enter specific keywords you want to track (optional but recommended)
    
    ### Step 2: Configure Analysis (Optional)
    - **Sidebar Settings**: Adjust TF-IDF parameters for fine-tuned analysis
    - **Keyword Count**: Change how many top keywords to display
    - **Bigrams**: Include two-word phrases for deeper analysis
    
    ### Step 3: Run Analysis
    - Click "Analyze Similarity" to start the analysis
    - Wait for processing (usually takes a few seconds)
    
    ## 📊 Understanding Your Results
    
    ### Similarity Scores
    - **0-30%**: Very different content - unique positioning
    - **30-60%**: Moderate similarity - balanced approach
    - **60%+**: High similarity - consider differentiation
    
    ### Keyword Analysis
    - **Your Top Keywords**: Most important terms in your content
    - **Keyword Overlap**: Shared terms with competitors
    - **Unique Keywords**: Terms competitors use that you don't
    
    ### Seed Keywords Performance
    - **Radar Chart**: Visual comparison of keyword usage
    - **Scores**: Higher scores mean better keyword optimization
    - **Recommendations**: Suggestions for improvement
    
    ## 💡 How to Gain Actionable Insights
    
    ### Content Strategy Insights
    1. **High Similarity (60%+)**:
       - Your content is too similar to competitors
       - **Action**: Develop unique value propositions
       - **Focus**: Create original content highlighting your strengths
    
    2. **Moderate Similarity (30-60%)**:
       - Good balance between relevance and uniqueness
       - **Action**: Maintain current strategy, fine-tune messaging
       - **Focus**: Strengthen areas where you're underperforming
    
    3. **Low Similarity (0-30%)**:
       - Highly unique positioning
       - **Action**: Ensure you're not missing industry-relevant terms
       - **Focus**: Add relevant keywords while maintaining uniqueness
    
    ### Competitive Analysis Insights
    - **Most Similar Competitor**: Study their strategy closely - they're your direct competition
    - **Least Similar Competitor**: Learn from their different approach
    - **Keyword Gaps**: Terms they use effectively that you're missing
    
    ### SEO and Content Optimization
    1. **Underperforming Seed Keywords**: 
       - Add more content around these terms
       - Improve keyword density naturally
    
    2. **Content Expansion Opportunities**:
       - Look at competitors' unique keywords
       - Identify content topics you're missing
    
    3. **Differentiation Strategies**:
       - Find gaps in competitor content
       - Develop content for underserved topics
    
    ## 📁 Using Your Downloaded Reports
    
    ### Markdown Report (.md)
    - **Use for**: Stakeholder presentations, documentation
    - **Contains**: Complete analysis with methodology
    
    ### CSV Data (.csv)
    - **Use for**: Further analysis in Excel/Google Sheets
    - **Contains**: Raw similarity scores and keyword data
    
    ### JSON Data (.json)
    - **Use for**: Integration with other tools, API consumption
    - **Contains**: Structured data for programmatic use
    
    ## 🎯 Best Practices
    
    1. **Content Quality**: Use substantial text (100+ words per competitor)
    2. **Representative Content**: Use homepage or key landing page content
    3. **Regular Analysis**: Re-analyze monthly to track improvements
    4. **Seed Keywords**: Choose 5-10 relevant industry terms
    5. **Action Items**: Implement 1-2 recommendations at a time
    
    ## ⚠️ Common Mistakes to Avoid
    
    - **Too Little Text**: Analysis needs sufficient content to be meaningful
    - **Mixed Content Types**: Don't mix product descriptions with blog posts
    - **Ignoring Context**: Consider your target audience when interpreting results
    - **Over-optimization**: Don't sacrifice readability for keyword density
    """)

# Usage Tips Alert
st.info("""
💡 **Pro Tip**: Start with the sample data to understand how the tool works, then use your own content. 
Focus on your homepage or key landing page content for the most relevant insights.
""")

# Sample data for demonstration
SAMPLE_DATA = {
    "competitor1": """We are a leading digital marketing agency specializing in SEO, social media marketing, and content creation. Our team of experts helps businesses grow their online presence through data-driven strategies. We offer comprehensive marketing solutions including website design, email marketing, and paid advertising campaigns.""",
    
    "competitor2": """Digital marketing experts providing SEO services, social media management, and online advertising solutions. We help companies increase their web traffic and conversions through proven marketing strategies. Our services include content marketing, website optimization, and brand development.""",
    
    "competitor3": """Professional marketing agency offering search engine optimization, social media campaigns, and digital advertising services. We specialize in helping businesses improve their online visibility and drive more qualified leads through strategic marketing initiatives.""",
    
    "own": """Innovative digital marketing solutions for modern businesses. We provide expert SEO services, social media strategy, and creative content development. Our approach combines data analytics with creative marketing to deliver measurable results for our clients' online growth."""
}

# Create input sections
st.header("📝 Content Input")
st.markdown("Please paste the text content from each website below:")

# Initialize session state for data
if 'comp1_data' not in st.session_state:
    st.session_state.comp1_data = ""
if 'comp2_data' not in st.session_state:
    st.session_state.comp2_data = ""
if 'comp3_data' not in st.session_state:
    st.session_state.comp3_data = ""
if 'own_data' not in st.session_state:
    st.session_state.own_data = ""
if 'seed_data' not in st.session_state:
    st.session_state.seed_data = ""

# Sample data loader
col_load, col_clear = st.columns([1, 1])
with col_load:
    if st.button("📋 Load Sample Data", help="Load example content to see how the analysis works"):
        st.session_state.comp1_data = SAMPLE_DATA["competitor1"]
        st.session_state.comp2_data = SAMPLE_DATA["competitor2"]
        st.session_state.comp3_data = SAMPLE_DATA["competitor3"]
        st.session_state.own_data = SAMPLE_DATA["own"]
        st.session_state.seed_data = "SEO, social media marketing, digital marketing, content creation"
        st.rerun()
        
with col_clear:
    if st.button("🗑️ Clear All", help="Clear all input fields"):
        st.session_state.comp1_data = ""
        st.session_state.comp2_data = ""
        st.session_state.comp3_data = ""
        st.session_state.own_data = ""
        st.session_state.seed_data = ""
        st.rerun()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Competitor Content")
    comp1 = st.text_area(
        "Competitor 1 Content",
        height=150,
        placeholder="Paste the text content from competitor 1's website here...",
        value=st.session_state.comp1_data,
        key="comp1_input"
    )
    
    comp2 = st.text_area(
        "Competitor 2 Content", 
        height=150,
        placeholder="Paste the text content from competitor 2's website here...",
        value=st.session_state.comp2_data,
        key="comp2_input"
    )
    
    comp3 = st.text_area(
        "Competitor 3 Content",
        height=150, 
        placeholder="Paste the text content from competitor 3's website here...",
        value=st.session_state.comp3_data,
        key="comp3_input"
    )

with col2:
    st.subheader("Your Website Content")
    own = st.text_area(
        "Your Website Content",
        height=450,
        placeholder="Paste your website's text content here...",
        value=st.session_state.own_data,
        key="own_input"
    )

# Seed keywords input
st.subheader("🎯 Seed Keywords (Optional)")
st.markdown("Enter specific keywords you want to focus on, separated by commas. These will get extra attention in the analysis:")
seed_keywords = st.text_input(
    "Seed Keywords",
    placeholder="e.g., digital marketing, SEO, social media, content strategy",
    help="Enter keywords separated by commas. The analysis will highlight these keywords and show their importance.",
    value=st.session_state.seed_data,
    key="seed_input"
)

# Analysis button
st.markdown("---")
if st.button("🔍 Analyze Similarity", type="primary", use_container_width=True):
    # Collect all texts
    texts = [comp1, comp2, comp3, own]
    
    # Validation
    if any(len(t.strip()) == 0 for t in texts):
        st.error("⚠️ Please fill in all four content boxes before analyzing.")
    else:
        with st.spinner("Analyzing content similarity and extracting keywords..."):
            # Preprocess texts
            processed_texts = [preprocess_text(text) for text in texts]
            
            # Check if processed texts are valid
            if any(len(t.strip()) == 0 for t in processed_texts):
                st.error("⚠️ After preprocessing, some content appears to be empty. Please ensure you have meaningful text content.")
            else:
                try:
                    # Vectorize using TF-IDF with dynamic settings
                    vectorizer = TfidfVectorizer(
                        stop_words='english',
                        max_features=max_features,
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_df=max_df
                    )
                    
                    tfidf_matrix = vectorizer.fit_transform(processed_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Calculate cosine similarity of own website vs competitors
                    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
                    
                    # Analyze seed keywords if provided
                    seed_analysis = None
                    if seed_keywords.strip():
                        seed_keywords_list = [kw.strip() for kw in seed_keywords.split(',')]
                        seed_analysis = analyze_seed_keywords(processed_texts, vectorizer, feature_names, seed_keywords_list)
                    
                    # Results Section
                    st.success("✅ Analysis Complete!")
                    
                    # Display similarity scores
                    st.header("📈 Similarity Scores")
                    
                    # Contextual help for similarity scores
                    with st.expander("ℹ️ How to Interpret Similarity Scores", expanded=False):
                        st.markdown("""
                        **Similarity Score Guide:**
                        - **0-30%**: Very different content - unique positioning (good for differentiation)
                        - **30-60%**: Moderate similarity - balanced approach (industry-relevant but distinct)
                        - **60-80%**: High similarity - consider differentiation (overlapping messaging)
                        - **80%+**: Very high similarity - urgent need for unique content
                        
                        **What This Means for Your Business:**
                        - Higher scores indicate more competitive overlap
                        - Lower scores suggest better differentiation
                        - Ideal range depends on your market position
                        """)
                    
                    st.markdown("""
                    Similarity scores range from 0 to 1, where:
                    - **0.0** = No similarity (completely different content)
                    - **0.5** = Moderate similarity (some common themes/terms)
                    - **1.0** = Identical content
                    """)
                    
                    # Create metrics display
                    col1, col2, col3 = st.columns(3)
                    
                    competitor_names = ["Competitor 1", "Competitor 2", "Competitor 3"]
                    colors = ["🔴", "🟠", "🟢"]
                    
                    for i, (score, name, color) in enumerate(zip(similarity_scores, competitor_names, colors)):
                        with [col1, col2, col3][i]:
                            st.metric(
                                label=f"{color} {name}",
                                value=f"{score:.1%}",
                                delta=f"{score:.3f} similarity score"
                            )
                    
                    # Interactive Similarity Chart
                    st.subheader("📊 Interactive Similarity Visualization")
                    similarity_chart = create_similarity_chart(similarity_scores, competitor_names)
                    st.plotly_chart(similarity_chart, use_container_width=True)
                    
                    # Seed Keywords Analysis (if provided)
                    if seed_analysis:
                        st.header("🎯 Seed Keywords Analysis")
                        st.markdown("Here's how your specified seed keywords perform across all content:")
                        
                        if seed_analysis:
                            seed_df = pd.DataFrame(seed_analysis)
                            seed_df.columns = ['Keyword', 'Your Score', 'Competitor 1', 'Competitor 2', 'Competitor 3', 'Total Importance']
                            
                            # Round scores for better display
                            for col in ['Your Score', 'Competitor 1', 'Competitor 2', 'Competitor 3', 'Total Importance']:
                                seed_df[col] = seed_df[col].round(3)
                            
                            st.dataframe(seed_df, use_container_width=True)
                            
                            # Insights about seed keywords
                            best_seed = seed_analysis[0] if seed_analysis else None
                            if best_seed:
                                st.info(f"🏆 **Top performing seed keyword:** '{best_seed['keyword']}' with total importance of {best_seed['total_score']:.3f}")
                                
                                if best_seed['your_score'] > 0:
                                    st.success(f"✅ You're using '{best_seed['keyword']}' effectively in your content!")
                                else:
                                    st.warning(f"⚠️ Consider adding more content about '{best_seed['keyword']}' to your website.")
                        else:
                            st.info("Your seed keywords weren't found in the content. Try using different keywords or check spelling.")
                        
                        # Radar Chart for Seed Keywords
                        if seed_analysis and len(seed_analysis) >= 2:
                            st.subheader("🎯 Seed Keywords Performance Radar")
                            radar_chart = create_keyword_radar_chart(seed_analysis)
                            if radar_chart:
                                st.plotly_chart(radar_chart, use_container_width=True)
                        
                        st.markdown("---")
                    
                    # Keyword Analysis Section
                    st.header("🔍 Keyword Overlap Analysis")
                    
                    with st.expander("💡 Understanding Keyword Analysis", expanded=False):
                        st.markdown("""
                        **How Keyword Analysis Helps:**
                        - **Your Top Keywords**: Shows what your content emphasizes most
                        - **Keyword Overlap**: Reveals shared terms with competitors (competitive themes)
                        - **Unique Keywords**: Terms competitors use that you might be missing
                        - **Importance Scores**: Higher scores indicate more prominent terms in content
                        
                        **Strategic Insights:**
                        1. **High Overlap**: You're competing directly on these terms
                        2. **Low Overlap**: Opportunity for differentiation or gap filling
                        3. **Missing Keywords**: Consider if you should include competitor keywords
                        4. **Unique Terms**: Your competitive advantages and differentiators
                        """)
                    
                    st.markdown("This section shows the most important keywords and their overlap between your content and each competitor.")
                    
                    # Extract keywords for own content using dynamic count
                    own_keywords = extract_keywords(processed_texts[-1], vectorizer, feature_names, top_n=top_keywords_count)
                    
                    # Show own content keywords
                    st.subheader("📌 Your Top Keywords")
                    if own_keywords:
                        own_kw_df = pd.DataFrame(own_keywords)
                        own_kw_df.columns = ['Keyword', 'Importance Score']
                        own_kw_df['Importance Score'] = own_kw_df['Importance Score'].round(3)
                        st.dataframe(own_kw_df, use_container_width=True)
                    else:
                        st.info("No significant keywords found in your content.")
                    
                    # Analyze overlap with each competitor
                    st.subheader("🔄 Keyword Overlap with Competitors")
                    
                    for i, (comp_text, comp_name, similarity_score) in enumerate(zip(processed_texts[:-1], competitor_names, similarity_scores)):
                        with st.expander(f"{competitor_names[i]} - Similarity: {similarity_score:.1%}", expanded=True):
                            # Extract competitor keywords using dynamic count
                            comp_keywords = extract_keywords(comp_text, vectorizer, feature_names, top_n=top_keywords_count)
                            
                            if comp_keywords and own_keywords:
                                # Find overlap
                                overlap = find_keyword_overlap(own_keywords, comp_keywords)
                                
                                if overlap:
                                    st.markdown(f"**Found {len(overlap)} overlapping keywords:**")
                                    
                                    # Create overlap dataframe
                                    overlap_df = pd.DataFrame(overlap)
                                    overlap_df['Your Score'] = overlap_df['own_score'].round(3)
                                    overlap_df['Competitor Score'] = overlap_df['competitor_score'].round(3)
                                    overlap_df['Combined Importance'] = (overlap_df['own_score'] + overlap_df['competitor_score']).round(3)
                                    
                                    # Display overlap table
                                    display_df = overlap_df[['keyword', 'Your Score', 'Competitor Score', 'Combined Importance']].copy()
                                    display_df.columns = ['Keyword', 'Your Score', 'Competitor Score', 'Combined Importance']
                                    st.dataframe(display_df, use_container_width=True)
                                    
                                    # Show competitor's unique keywords
                                    comp_unique = set([kw[0] for kw in comp_keywords]) - set([kw[0] for kw in own_keywords])
                                    if comp_unique:
                                        st.markdown(f"**{comp_name}'s unique keywords you might consider:**")
                                        unique_keywords = [kw for kw in comp_keywords if kw[0] in comp_unique][:5]
                                        unique_kw_text = ", ".join([f"*{kw[0]}*" for kw in unique_keywords])
                                        st.markdown(unique_kw_text)
                                else:
                                    st.info(f"No keyword overlap found with {comp_name}.")
                            else:
                                st.warning(f"Unable to extract meaningful keywords from {comp_name}'s content.")
                    
                    # Summary and Recommendations
                    st.header("💡 Summary & Recommendations")
                    
                    # Find highest and lowest similarity
                    max_sim_idx = similarity_scores.argmax()
                    min_sim_idx = similarity_scores.argmin()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"**Most Similar:** {competitor_names[max_sim_idx]} ({similarity_scores[max_sim_idx]:.1%})")
                        if similarity_scores[max_sim_idx] > 0.5:
                            st.markdown("This competitor has very similar content to yours. Consider differentiating your messaging.")
                        
                    with col2:
                        st.info(f"**Least Similar:** {competitor_names[min_sim_idx]} ({similarity_scores[min_sim_idx]:.1%})")
                        if similarity_scores[min_sim_idx] < 0.3:
                            st.markdown("This competitor targets a different market segment or uses different messaging strategies.")
                    
                    # Overall recommendations
                    avg_similarity = similarity_scores.mean()
                    st.markdown("### Overall Analysis:")
                    
                    # Define interpretation for later use
                    if avg_similarity > 0.6:
                        interpretation = "HIGH"
                        st.warning(f"**High Average Similarity ({avg_similarity:.1%}):** Your content is very similar to competitors. Consider developing more unique value propositions and messaging.")
                    elif avg_similarity > 0.3:
                        interpretation = "MODERATE"
                        st.info(f"**Moderate Average Similarity ({avg_similarity:.1%}):** Good balance between industry relevance and unique positioning.")
                    else:
                        interpretation = "LOW"
                        st.success(f"**Low Average Similarity ({avg_similarity:.1%}):** Your content is quite unique compared to competitors, which can be an advantage for differentiation.")
                    
                    # Download Section
                    st.header("📥 Export & Download")
                    st.markdown("Download your analysis results in different formats:")
                    
                    col_download1, col_download2, col_download3 = st.columns(3)
                    
                    # Generate comprehensive report
                    overlap_data = []  # Placeholder for now
                    comprehensive_report = generate_comprehensive_report(
                        similarity_scores, competitor_names, own_keywords, seed_analysis, overlap_data
                    )
                    
                    with col_download1:
                        # Download comprehensive report as markdown
                        st.download_button(
                            label="📄 Download Full Report (MD)",
                            data=comprehensive_report,
                            file_name=f"similarity_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            help="Download a comprehensive markdown report with all analysis results"
                        )
                    
                    with col_download2:
                        # Create CSV data for similarity scores
                        csv_data = pd.DataFrame({
                            'Competitor': competitor_names,
                            'Similarity_Score': similarity_scores,
                            'Similarity_Percentage': [f"{score:.1%}" for score in similarity_scores]
                        })
                        
                        # Add keyword data if available
                        if own_keywords:
                            keyword_df = pd.DataFrame(own_keywords[:10])
                            keyword_df.columns = ['Keyword', 'Importance_Score']
                            
                        csv_buffer = io.StringIO()
                        csv_data.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📊 Download Data (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"similarity_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download similarity scores and data in CSV format"
                        )
                    
                    with col_download3:
                        # Create JSON export with all data
                        json_data = {
                            'analysis_date': datetime.now().isoformat(),
                            'similarity_scores': {
                                competitor_names[i]: float(similarity_scores[i]) 
                                for i in range(len(competitor_names))
                            },
                            'average_similarity': float(avg_similarity),
                            'top_keywords': [{'keyword': kw[0], 'score': float(kw[1])} for kw in own_keywords[:15]] if own_keywords else [],
                            'seed_analysis': seed_analysis if seed_analysis else [],
                            'recommendations': {
                                'most_similar_competitor': competitor_names[max_sim_idx],
                                'least_similar_competitor': competitor_names[min_sim_idx],
                                'similarity_level': interpretation
                            }
                        }
                        
                        st.download_button(
                            label="🔗 Download JSON Data",
                            data=json.dumps(json_data, indent=2),
                            file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download complete analysis data in JSON format for further processing"
                        )
                    
                    # Action Items Section
                    st.header("✅ Next Steps & Action Items")
                    
                    action_items = []
                    
                    if avg_similarity > 0.6:
                        action_items.append("🎯 **High Priority**: Develop unique value propositions to differentiate from competitors")
                        action_items.append("📝 **Content Strategy**: Create original content that highlights your unique strengths")
                    
                    if seed_analysis:
                        underperforming_seeds = [item for item in seed_analysis if item['your_score'] < 0.05]
                        if underperforming_seeds:
                            action_items.append(f"🔍 **SEO Focus**: Strengthen content around: {', '.join([item['keyword'] for item in underperforming_seeds[:3]])}")
                    
                    # Best competitor analysis
                    best_competitor_idx = similarity_scores.argmax()
                    action_items.append(f"🕵️ **Competitive Analysis**: Study {competitor_names[best_competitor_idx]}'s content strategy more closely")
                    
                    # Keyword opportunities
                    if own_keywords and len(own_keywords) < 10:
                        action_items.append("📈 **Content Expansion**: Your content could benefit from more diverse keyword coverage")
                    
                    for i, item in enumerate(action_items, 1):
                        st.markdown(f"{i}. {item}")
                    
                    # Timeline suggestion
                    st.info("💡 **Suggested Timeline**: Review and implement these changes over the next 2-4 weeks, then re-analyze to track improvements.")
                    
                    # Quick Insights Summary
                    st.header("🎯 Quick Insights Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Key Findings")
                        st.markdown(f"• Average similarity: **{avg_similarity:.1%}**")
                        st.markdown(f"• Most similar competitor: **{competitor_names[max_sim_idx]}**")
                        st.markdown(f"• Your top keyword: **{own_keywords[0][0] if own_keywords else 'N/A'}**")
                        
                        if seed_analysis:
                            best_seed = seed_analysis[0]
                            st.markdown(f"• Best performing seed keyword: **{best_seed['keyword']}**")
                    
                    with col2:
                        st.markdown("### 🚀 Priority Actions")
                        priority_actions = []
                        
                        if avg_similarity > 0.6:
                            priority_actions.append("🔴 **HIGH**: Differentiate content immediately")
                        elif avg_similarity < 0.3:
                            priority_actions.append("🟡 **MEDIUM**: Add industry-relevant keywords")
                        else:
                            priority_actions.append("🟢 **LOW**: Fine-tune existing strategy")
                            
                        if seed_analysis:
                            underperforming = [item for item in seed_analysis if item['your_score'] < 0.05]
                            if underperforming:
                                priority_actions.append(f"🔍 **SEO**: Focus on {underperforming[0]['keyword']}")
                        
                        for action in priority_actions:
                            st.markdown(f"• {action}")
                    
                    # Call-to-Action
                    st.markdown("---")
                    st.markdown("### 📈 Ready to Improve Your Content?")
                    col_cta1, col_cta2, col_cta3 = st.columns(3)
                    
                    with col_cta1:
                        st.markdown("**📄 Download Report**")
                        st.markdown("Share insights with your team")
                    
                    with col_cta2:
                        st.markdown("**🔄 Re-analyze**")
                        st.markdown("Check progress after changes")
                    
                    with col_cta3:
                        st.markdown("**⚙️ Adjust Settings**")
                        st.markdown("Fine-tune analysis parameters")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {str(e)}")
                    st.markdown("Please check that your content contains meaningful text and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
        This tool uses TF-IDF vectorization and cosine similarity to analyze content similarity.<br>
        Results are based on textual content analysis and keyword frequency patterns.
    </small>
</div>
""", unsafe_allow_html=True)
