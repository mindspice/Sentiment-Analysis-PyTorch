import os
import pandas as pd
from pathlib import Path
from processing import PreProcessor
from sentiment_analysis import SentimentAnalyzer

sentiment_analyzer = SentimentAnalyzer("models/vocab.pt", "models/sentiment_model.pt")
pre_processor = PreProcessor()


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def home_menu():
    clear_screen()
    print("1. Analyze Single Text")
    print("2. Analyze File")
    print("3. Evaluate Model")
    print("4. Exit")
    return input("Please choose an option: ")


def analyze_single_text():
    df = pd.DataFrame(columns=['text', 'sentiment'])
    while True:
        clear_screen()
        print(f"Analyzed texts so far: {len(df)}")
        print(f"Lowest Sentiment score: {round(df['sentiment'].min(), 3)}")
        print(f"Highest Sentiment score: {round(df['sentiment'].max(), 3)}")
        print(f"Average Sentiment score: {round(df['sentiment'].mean(), 3)}", "\n")
        text = input("\nEnter the text to be analyzed: ")

        sentiment_score = sentiment_analyzer.predict_sentiment(text)
        df.loc[len(df)] = [text, sentiment_score]

        while True:
            clear_screen()
            print(f"Original text: {text}")
            print(f"Preprocessed text: {pre_processor.preprocess_text(text)}")
            print(f"Sentiment score: {sentiment_score}", "\n")
            print(f"Analyzed texts so far: {len(df)}")
            print(f"Analyzed texts so far: {len(df)}")
            print(f"Lowest Sentiment score: {round(df['sentiment'].min(), 2)}")
            print(f"Highest Sentiment score: {round(df['sentiment'].max(), 2)}")
            print(f"Average Sentiment score: {round(df['sentiment'].mean(), 2)}", "\n")
            print("1. Analyze Another Text")
            print("2. Show Sentiment Distribution")
            print("3. Show Sentiment Classifications")
            print("4. Show Sentiment Statistics")
            print("5. Export Predictions")
            print("6. Exit To Main Menu")
            option = input("\nPlease choose an option: ")

            if option == "1":
                break
            elif option == "2":
                sentiment_analyzer.plot_sentiment_distribution(df)
            elif option == "3":
                sentiment_analyzer.plot_sentiment_classifications(df)
            elif option == "4":
                sentiment_analyzer.plot_statistics(df)
            elif option == "5":
                filename = input("\nEnter filename prefix: ")
                sentiment_analyzer.write_sentiments_to_csv(df, filename)
            elif option == "6":
                return
            else:
                print("Invalid option. Please try again.")


def analyze_text_from_file():
    clear_screen()
    while True:
        file_name = input("Enter the filename (in the 'data/' directory): ")
        file_name = Path(f"data/{file_name}")
        if not os.path.isfile(file_name):
            print("File does not exist. Please try again.")
            continue

        print("Analyzing...")
        df = sentiment_analyzer.predict_sentiments_from_file(file_name)

        while True:
            clear_screen()
            print(f"Analyzed texts: {len(df)}")
            print(f"Analyzed texts so far: {len(df)}")
            print(f"Lowest Sentiment score: {round(df['sentiment'].min(), 3)}")
            print(f"Highest Sentiment score: {round(df['sentiment'].max(), 3)}")
            print(f"Average Sentiment score: {round(df['sentiment'].mean(), 3)}", "\n")
            print("1. Show Sentiment Distribution")
            print("2. Show Sentiment Classifications")
            print("3. Show Sentiment Statistics")
            print("4. Analyze Another File")
            print("5. Export Predictions")
            print("6. Exit To Main Menu")
            option = input("\nPlease choose an option: ")

            if option == "1":
                sentiment_analyzer.plot_sentiment_distribution(df)
            elif option == "2":
                sentiment_analyzer.plot_sentiment_classifications(df)
            elif option == "3":
                sentiment_analyzer.plot_statistics(df)
            elif option == "4":
                break
            elif option == "5":
                filename = input("\nEnter filename prefix: ")
                sentiment_analyzer.write_sentiments_to_csv(df, filename)
            elif option == "6":
                return
            else:
                print("Invalid option. Please try again.")


def evaluate_model():
    # Prompt for sample fraction and validate it
    while True:
        file_name = input("Enter file name to evaluate (in the 'test/' directory: ")
        file_name = Path(f"test/{file_name}")
        if not os.path.isfile(file_name):
            print("File does not exist. Please try again.")
            continue

        while True:
            print("Consider using a value of 0.05-0.1 if evaluating against a large test set and running via CPU")
            sample_fraction = input("Enter sample fraction (between 0 and 1): ")
            try:
                sample_fraction = float(sample_fraction)
                if 0 <= sample_fraction <= 1:
                    break
                else:
                    print("Sample fraction must be between 0 and 1. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a numerical value.")

    # Prompt for random seed and validate it
    while True:
        random_seed = input("Enter random seed (integer): ")
        try:
            random_seed = int(random_seed)
            break
        except ValueError:
            print("Invalid input. Please enter an numerical value.")

    print("Evaluating...\n")
    accuracies, accuracy = sentiment_analyzer.eval(file_name, sample_fraction=sample_fraction, random_seed=random_seed)


    while True:
        clear_screen()
        print(f"Accuracy: {accuracy:.2%}")
        print("\n1. Plot accuracy Trend")
        print("2. Re-evaluate")
        print("3. Exit To Main Menu")
        option = input("\nPlease choose an option: ")

        if option == "1":
            sentiment_analyzer.plot_accuracies(accuracies)
        elif option == "2":
            evaluate_model()
        elif option == "3":
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    clear_screen()
    while True:
        option = home_menu()

        if option == "1":
            analyze_single_text()
        elif option == "2":
            analyze_text_from_file()
        elif option == "3":
            evaluate_model()
        elif option == "4":
            print("Goodbye!")
            break
        else:
            print("\nInvalid option. Please try again.")
