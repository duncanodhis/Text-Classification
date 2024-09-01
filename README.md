
---

# Text Classification Using Naive Bayes

This project demonstrates a simple text classification model using the Naive Bayes algorithm. The model classifies text into categories (such as `positive`, `negative`, and `neutral`) based on labeled data. The classification process includes text preprocessing (such as tokenization, stop-word removal, and TF-IDF transformation), training a Naive Bayes model, and evaluating its accuracy.

## Features
- Preprocessing of text data (lowercasing, punctuation removal, stop-word filtering).
- Classification of text into predefined categories using a Naive Bayes classifier.
- Evaluation of model performance using accuracy and classification reports.
- Prediction of sentiment for new, unseen text inputs.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/duncanodhis/text-classification.git
   ```

2. Navigate into the project directory:

   ```bash
   cd text-classification
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install the dependencies manually:

   ```bash
   pip install scikit-learn pandas numpy nltk
   ```

## Usage

1. **Run the script**:

   To run the text classification model, simply execute the script:

   ```bash
   python text_classification.py
   ```

2. **Input Data**:

   The dataset consists of a set of texts and corresponding labels. In this example, a small dataset is provided within the script. You can replace this with your own dataset.

3. **Preprocessing**:

   The script preprocesses text by:
   - Converting it to lowercase.
   - Removing punctuation.
   - Removing stop words using NLTK's list of English stop words.

4. **Train-Test Split**:

   The data is split into training (80%) and testing (20%) sets using Scikit-learn’s `train_test_split` method.

5. **Model Training**:

   The `MultinomialNB` classifier is trained on the preprocessed text data, where:
   - `CountVectorizer` is used to convert text into token counts.
   - `TfidfTransformer` is used to calculate the TF-IDF values from the token counts.

6. **Evaluation**:

   After training, the model is evaluated using accuracy and classification metrics (precision, recall, and F1-score).

7. **Prediction**:

   You can input new text to the model, and it will predict the category (e.g., positive, negative).

### Example

You can test the model by modifying the input text directly in the script under `new_text`.

```python
new_text = ["The product quality is excellent!"]
```

The model will predict the sentiment of the input text and output the predicted category.

### Sample Output

```bash
Accuracy: 100.00%

Classification Report:
               precision    recall  f1-score   support

    negative       1.00      1.00      1.00         1
     neutral       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Predicted Sentiment: positive
```

## Customization

- **Dataset**: Replace the `data` dictionary in the script with your own text and label dataset.
- **Classifiers**: You can try other classifiers like SVM (`LinearSVC`) or Logistic Regression by replacing the `MultinomialNB` classifier in the pipeline.
- **Preprocessing**: Add more preprocessing techniques such as stemming, lemmatization, or other feature extraction methods.

## Dependencies

- Python 3.x
- Scikit-learn
- Pandas
- Numpy
- NLTK

To install NLTK's stopwords, use:

```python
import nltk
nltk.download('stopwords')
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing

Feel free to submit a pull request if you'd like to contribute to this project or add more features.

## Contact

For any inquiries or issues, please open an issue in the repository or contact `duncanabi.kenya@gmail.com`.

---

### Author

Developed by [Duncan Abonyo](https://github.com/duncanodhis).

---
