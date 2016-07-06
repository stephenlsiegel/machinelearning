def count_words(s, n):
    """Return the n most frequently occuring words in s."""
    
    # TODO: Count the number of occurences of each word in s
    wordList = s.split(' ')
    wordSet = set(s.split(' ')) # this is a set of unique words in the sentence s
    top_n = []
    
    for wordS in wordSet:
        
        word_count = 0
        
        for wordL in wordList:
            if wordL == wordS:
                word_count += 1
            else:
                continue
            
        word_tuple = (wordS, word_count)
        top_n.append(word_tuple)

    # TODO: Sort the occurences in descending order (alphabetically in case of ties)
    top_n.sort(key = lambda x: (-x[1],x[0]), reverse=False)
    
    # TODO: Return the top n words as a list of tuples (<word>, <count>)
    top_n = top_n[0:n]
    return top_n

print count_words("cat bat mat cat bat cat", 3)
print count_words("betty bought a bit of butter but the butter was bitter", 3)

# count_words("cat bat mat cat bat cat", 3)
# count_words("betty bought a bit of butter but the butter was bitter", 3)