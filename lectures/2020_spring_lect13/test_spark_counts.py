from pyspark import SparkContext
from operator import add # Required for reduceByKey
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
    sc = SparkContext("local", "wordcount")
    
    # creating a RDD from pg2701.txt
    text = sc.textFile('pg2701.txt')
    
    # creating a list of words in the RDD
    words = text.flatMap(splitter)
#     print("words:", words.take(10))
    
    # mapping the list of words in a tuple
    # make (key, value) pairs
    words_mapped = words.map(lambda w: (w, 1))
    
    #sort them
    sorted_map = words_mapped.sortByKey()
    
    #reduce by key i.e. count how many of the words have a given key
    counts = sorted_map.reduceByKey(add)
    
    print("counts: ", counts.take(500))
    
    
