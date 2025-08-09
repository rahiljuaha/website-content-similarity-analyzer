# Website Content Similarity Analyzer

A powerful Streamlit web application that analyzes content similarity between your website and competitors using advanced machine learning techniques.

## Features

### Core Analysis
- **Similarity Scoring**: TF-IDF vectorization with cosine similarity measurement
- **Keyword Extraction**: Automated identification of important terms and phrases
- **Competitive Analysis**: Compare your content against up to 3 competitors
- **Seed Keyword Tracking**: Monitor performance of specific keywords you care about

### Visualizations
- **Interactive Charts**: Bar charts for similarity scores
- **Radar Charts**: Visual comparison of seed keyword performance across all content
- **Metrics Dashboard**: Clear display of key performance indicators

### Advanced Features
- **Customizable Analysis**: Adjust TF-IDF parameters for fine-tuned results
- **Sample Data**: Pre-loaded examples to demonstrate functionality
- **Export Options**: Download reports in Markdown, CSV, and JSON formats
- **Actionable Insights**: Specific recommendations based on analysis results

### User Experience
- **Comprehensive Guide**: Step-by-step instructions and best practices
- **Contextual Help**: Pop-up explanations for key concepts
- **Professional Reports**: Detailed analysis with methodology and recommendations
- **Mobile-Friendly**: Responsive design that works on all devices

## How It Works

1. **Input Content**: Paste text from your website and up to 3 competitors
2. **Add Keywords**: Optionally specify seed keywords to track
3. **Configure Settings**: Adjust analysis parameters in the sidebar
4. **Run Analysis**: Get detailed similarity scores and keyword insights
5. **Download Reports**: Export results for sharing and further analysis

## Technology Stack

- **Frontend**: Streamlit for interactive web interface
- **Machine Learning**: Scikit-learn for TF-IDF vectorization and similarity computation
- **Data Processing**: Pandas for data manipulation and analysis
- **Visualizations**: Plotly for interactive charts and graphs
- **Text Processing**: Regular expressions and string manipulation for content cleaning


### For Local Development
1. Clone the repository
2. Install dependencies: `pip install -r dependencies.txt`
3. Run the app: `streamlit run app.py`

### Dependencies
- streamlit>=1.28.0
- scikit-learn>=1.3.0
- pandas>=2.1.0
- plotly>=5.15.0

## Usage Guide

### Quick Start
1. Click "Load Sample Data" to see how the tool works
2. Replace with your own content and competitors' content
3. Add relevant seed keywords (optional)
4. Click "Analyze Similarity"
5. Review results and download reports

### Understanding Results

#### Similarity Scores
- **0-30%**: Unique positioning (good for differentiation)
- **30-60%**: Balanced approach (industry-relevant but distinct)
- **60%+**: High similarity (consider differentiation strategies)

#### Keyword Analysis
- **Top Keywords**: Most important terms in your content
- **Overlap Analysis**: Shared terms with competitors
- **Unique Keywords**: Terms competitors use that you might be missing

### Best Practices
1. Use substantial text content (100+ words per competitor)
2. Focus on homepage or key landing page content
3. Choose 5-10 relevant seed keywords
4. Re-analyze monthly to track improvements
5. Implement 1-2 recommendations at a time


## Contributing

This project was built to help businesses understand their competitive positioning through content analysis. Feel free to suggest improvements or report issues.

## Use Cases

- **Content Marketing**: Identify gaps and opportunities in your content strategy
- **SEO Analysis**: Discover keywords you might be missing
- **Competitive Research**: Understand how your messaging compares to competitors
- **Brand Positioning**: Ensure your content differentiates your brand effectively
- **Content Planning**: Make data-driven decisions about future content

## Technical Details

### Analysis Methodology
1. **Text Preprocessing**: Lowercase conversion, punctuation removal, whitespace normalization
2. **Vectorization**: TF-IDF with configurable n-gram ranges and frequency thresholds
3. **Similarity Computation**: Cosine similarity between document vectors
4. **Keyword Extraction**: TF-IDF score ranking with configurable result counts

### Configuration Options
- Maximum features (500-5000)
- Document frequency thresholds
- N-gram ranges (unigrams, bigrams)
- Top keywords display count

## License

This project is designed for educational and business analysis purposes. Please ensure you have permission to analyze any content you input into the tool.
