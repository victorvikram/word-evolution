"""
function for counting tuples using sparse arrays, was very slow
"""
for j, speech in enumerate(speech_lists):
        if j % 10 == 0:
            print(f"{wind}: {j}", end="\r")
        
        speech = [sorted_word_dict[word] for word in speech if word in sorted_word_dict]
        tupleArr = KTuples.findKTuplesFrameLst(n, speech, frame, sparseArr=True)
        dim = tupleArr.coords.shape[0]

        focalMask = (tupleArr.coords < num_focal_words).any(axis=0)
        contextMask = (tupleArr.coords < num_context_words).all(axis=0)

        mask = (focalMask & contextMask)
        indices = np.where(mask)[0]
        coords = tupleArr.coords[:,indices]
        data = tupleArr.data[indices]
        filteredArr = sparse.COO(coords=coords, data=data, shape=(num_context_words,)*dim)
        totalArr += filteredArr