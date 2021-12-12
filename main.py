import gensim.downloader
import csv


def do_job(model_name):
    model = gensim.downloader.load(model_name)

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
            if reader[i][0] in model and (
                    reader[i][2] in model or reader[i][3] in model or reader[i][4] in model or
                    reader[i][
                        5] in model):
                label = "correct"
                correct_count += 1
            else:
                label = "guess"
                guess_count += 1
            for j in range(2, len(reader[i])):
                try:
                    if max_similarity < model.similarity(reader[i][0], reader[i][j]):
                        guess = reader[i][j]
                        max_similarity = model.similarity(reader[i][0], reader[i][j])
                except:
                    pass

            if label != "guess" and guess != reader[i][1]:  # reader[i][1] answer
                label = "wrong"
                correct_count -= 1
            analysis.append([reader[i][0] + "," + reader[i][1] + "," + guess + "," + label])
    with open(model_name + '-details.csv', 'w', newline='') as result:
        result_writer = csv.writer(result)
        result_writer.writerows(analysis)

    corpus_model = len(model)
    accuracy_model = correct_count / (80 - guess_count)
    total_model = 80 - guess_count

    return model_name + "," + str(corpus_model) + "," + str(correct_count) + "," + str(
        total_model) + "," + str(accuracy_model)


if __name__ == '__main__':
    print(list(gensim.downloader.info()['models'].keys()))

    wiki_300 = do_job('fasttext-wiki-news-subwords-300')
    # task 2.1
    twitter_200 = do_job('glove-twitter-200')
    giga_word_200 = do_job('glove-wiki-gigaword-200')

    with open('analysis.csv', 'w', newline='') as analysis_file:
        wr = csv.writer(analysis_file)
        wr.writerow([[wiki_300]])
        wr.writerow([[twitter_200]])
        wr.writerow([[giga_word_200]])
