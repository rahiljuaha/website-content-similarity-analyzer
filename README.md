# 📊 Website Content Similarity Analyzer

A powerful Streamlit app that compares your website content with up to three competitors using advanced Natural Language Processing (NLP) techniques. Instantly see similarity scores, keyword overlaps, competitor gaps, and get actionable recommendations to improve your content strategy.

---

## 🚀 Features

- **Content Similarity Analysis:**  
  Calculates similarity between your site and competitors using TF-IDF vectorization and cosine similarity.

- **Top Keyword Extraction:**  
  Identifies and ranks your most important keywords.

- **Keyword Overlap & Gap Analysis:**  
  Shows which keywords you share with competitors and what you're missing.

- **Seed Keyword Tracking:**  
  Track your performance across hand-picked "seed" keywords.

- **Interactive Visualizations:**  
  Includes bar and radar charts for at-a-glance insights.

- **Comprehensive Export:**  
  Download your analysis as Markdown, CSV, or JSON.

- **No Coding Required:**  
  All analysis done via a friendly Streamlit interface.

---

## 🖥️ How to Use

1. **Install Requirements**

   ```
   pip install -r requirements.txt
   ```

2. **Run the App**

   ```
   streamlit run app.py
   ```

3. **Input Content**

   - Paste your website content and up to three competitors' content.
   - (Optional) Add "seed keywords" you want to track (comma-separated).

4. **Configure Analysis (Optional)**

   - Adjust TF-IDF parameters in the sidebar for deeper or broader keyword analysis.
   - Choose to include bigrams (two-word phrases).

5. **Analyze & View Results**

   - Click "Analyze Similarity".
   - View similarity metrics, keyword tables, overlap analysis, and radar charts.

6. **Download Your Results**

   - Export your findings as Markdown, CSV, or JSON for further analysis or sharing.

---

## 📋 Example Use Cases

- **Competitive SEO Benchmarking**
- **Content Strategy Audits**
- **Web Copy Differentiation**
- **Market Positioning Research**
- **Gap Analysis for Content Development**

---

## 📦 Requirements

- Python 3.7+
- [`streamlit`](https://streamlit.io/)
- `scikit-learn`
- `pandas`
- `plotly`

Install all dependencies via:

```
pip install -r requirements.txt
```

---

## 🧑‍💻 Sample Data Included

Click "Load Sample Data" in the app to see a ready-to-run demo with example marketing agency content.

---

## 📝 Methodology

- **Preprocessing:** Lowercasing, punctuation removal, stopword filtering.
- **Vectorization:** TF-IDF (unigram/bigram, customizable).
- **Similarity:** Cosine similarity on TF-IDF vectors.
- **Keyword Extraction:** Ranked by TF-IDF importance.
- **Visualization:** Plotly charts for clarity.

---

## 📄 License

MIT License.  
Free to use, modify, and distribute. Attribution appreciated!

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for rapid prototyping.
- Powered by [scikit-learn](https://scikit-learn.org/) and [Plotly](https://plotly.com/).
- Inspired by real-world needs for content differentiation and SEO strategy.

---

## 💡 Tips

- Use substantial, representative text (100+ words per competitor).
- Compare homepage or key landing page content for best insights.
- Re-analyze regularly to track improvement.

---

**Ready to get started?**  
Run `streamlit run app.py` and discover how your site stacks up!
