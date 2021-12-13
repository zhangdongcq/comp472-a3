import gensim.downloader
import csv
import random
import pandas as pd
import matplotlib.pyplot as plt


def do_job(model_name):
    model_local = gensim.downloader.load(model_name)
    analysis = []
    correct_count = 0
    guess_count = 0
    with open('synonyms.csv', "r") as f:
        read = csv.reader(f)
        reader = list(read)
        # reader[i][0] question, reader[i][1] answer,
        for i in range(1, len(reader)):
            max_similarity = 0
            guess = reader[i][2]
            # either question-word or all four guess-words (or all 5 words) were not found in the embedding model
            if reader[i][0] in model_local and (
                    reader[i][2] in model_local or reader[i][3] in model_local or reader[i][4] in model_local or
                    reader[i][
                        5] in model_local):
                label_local = "correct"
                correct_count += 1
            else:
                label_local = "guess"
                guess_count += 1
            for j in range(2, len(reader[i])):
                try:
                    if max_similarity < model_local.similarity(reader[i][0], reader[i][j]):
                        guess = reader[i][j]
                        max_similarity = model_local.similarity(reader[i][0], reader[i][j])
                except:
                    pass

            if label_local != "guess" and guess != reader[i][1]:  # reader[i][1] answer
                label_local = "wrong"
                correct_count -= 1
            analysis.append([reader[i][0] + "," + reader[i][1] + "," + guess + "," + label_local])
    with open(model_name + '-details.csv', 'w', newline='') as result:
        result_writer = csv.writer(result)
        result_writer.writerows(analysis)

    corpus_model = len(model_local)
    accuracy_model = correct_count / (80 - guess_count)
    total_model = 80 - guess_count

    return model_name + "," + str(corpus_model) + "," + str(correct_count) + "," + str(
        total_model) + "," + str(accuracy_model)


if __name__ == '__main__':
    print(list(gensim.downloader.info()['models'].keys()))

    wiki_300 = do_job('fasttext-wiki-news-subwords-300')
    # task 2.1 -> 2 new models from different corpora [glove-twitter-200] and
    # [glove-wiki-gigaword-200] but same embedding size [200]
    twitter_200 = do_job('glove-twitter-200')
    giga_word_200 = do_job('glove-wiki-gigaword-200')

    # task 2.2 -> 2 new models from the same corpus [twitter] but different embedding size [25] and [50]
    twitter_25 = do_job('glove-twitter-25')
    twitter_50 = do_job('glove-twitter-50')
    with open('analysis.csv', 'w', newline='') as analysis_file:
        wr = csv.writer(analysis_file)
        wr.writerow([[wiki_300]])
        wr.writerow([[twitter_200]])
        wr.writerow([[giga_word_200]])
        wr.writerow([[twitter_25]])
        wr.writerow([[twitter_50]])

    data = pd.read_csv('synonyms.csv', sep=",", header=0)
    # random baseline
    random.seed(0)
    predictions = list()
    for row in data.iterrows():
        random_prediction = row[1][str(random.randint(1, 3))]
        label = 'correct' if random_prediction == row[1]["answer"] else 'wrong'
        s = F"{row[1]['question']},{row[1]['answer']},{random_prediction},{label}"
        predictions.append(s)

    with open(f"random-baseline-details.csv", 'w') as f:
        for prediction in predictions:
            f.write(prediction + '\n')

    # compare them to a random baseline and a human gold-standard
    with open(f"random-baseline-details.csv", 'r') as f:
        for line in f.readlines():
            predictions.append(line.split(','))
            predictions[-1][-1] = predictions[-1][-1].rstrip("\n")
    C = list(map(lambda x: x[-1], predictions)).count("correct")
    V = list(map(lambda x: x[-1], predictions)).count("wrong") + C
    random_baseline = "random-baseline" + "," + str(C / V)
    human_standard = "human-gold-standard" + "," + str(0.8557)

    with open('analysis_with_random_baseline.csv', 'w', newline='') as analysis_file:
        wr = csv.writer(analysis_file)
        wr.writerow([[wiki_300]])
        wr.writerow([[twitter_200]])
        wr.writerow([[giga_word_200]])
        wr.writerow([[twitter_25]])
        wr.writerow([[twitter_50]])
        wr.writerow([[random_baseline]])
        wr.writerow([[human_standard]])

    # Prepare and plot graphic
    with open('perform.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(['name', 'accuracy'])
        # write the data
        writer.writerow(['random-baseline', 0.2875])
        writer.writerow(['human-gold-standard', 0.8557])
        writer.writerow(['glove-twitter-50', 0.46153846153846156])
        writer.writerow(['glove-twitter-25', 0.46153846153846156])
        writer.writerow(['glove-wiki-gigaword-200', 0.85])
        writer.writerow(['glove-twitter-200', 0.5641025641025641])
        writer.writerow(['fasttext-wiki-news-subwords-300', 0.925])
    csv_file = 'perform.csv'
    data = pd.read_csv(csv_file)
    model = data["name"]
    ac = data["accuracy"]
    x = list(model)
    y = list(ac)
    plt.figure(figsize=(21, 12))
    plt.xlabel('Model Name')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of different models')
    plt.bar(x, y, color=['red', 'green'])
    plt.savefig('performance.pdf')
