import pandas as pd
import unittest
import sparse
import numpy as np
import context_distribs as cd
import process_speeches as ps
import context_wrangling as cw
import word_counting as wc
import load_data as ld
import distances as dist
import os

from numpy.testing import assert_array_almost_equal

import pickle as pkl

class TestContextDistribs(unittest.TestCase):
       def test_getDistribForWordI(self):
              coOccurTrip = sparse.COO(np.array([[1, 4, 3, 2, 2, 6, 7, 2, 5, 3, 5, 3, 0, 3, 3],
                                                 [0, 3, 2, 5, 3, 7, 9, 5, 5, 5, 3, 3, 3, 5, 3],
                                                 [3, 5, 7, 4, 4, 8, 3, 8, 4, 6, 2, 5, 3, 3, 3]]),
                                          data=np.array([4, 2, 3, 6, 7, 4, 3, 2, 6, 8, 6, 4, 2, 1, 4]))

              coOccurPair = sparse.COO(np.array([[0, 2, 2, 3, 5, 5, 3, 5, 2, 5, 7, 3, 5, 3, 3],
                                                        [3, 5, 8, 1, 4, 8, 4, 3, 3, 6, 3, 9, 3, 6, 3]]),
                                          data=np.array([6, 5, 3, 7, 8, 2, 5, 7, 8, 3, 1, 6, 9, 1, 2]))

              cd.getDistribForWordI(3, coOccurTrip, coOccurPair)

              coOccurTrip = sparse.COO(np.array([[1, 4, 3],
                                                 [0, 3, 2],
                                                 [3, 5, 6]]),
                                          data=np.array([1, 2, 3]))

              coOccurPair = sparse.COO(np.array([[0, 2, 3, 3, 4, 3, 3],
                                                 [3, 3, 1, 3, 3, 5, 6]]),
                                          data=np.array([1, 2, 3, 5, 5, 3, 5]))

              tripDistrib, indepTripDistrib = cd.getDistribForWordI(3, coOccurTrip, coOccurPair)
              expectedTripDistrib = np.array([1/6, 1/2, 1/3])
              expectedIndepTripDistrib = np.array([1/192, 5/288, 5/192])

              self.assertTrue((tripDistrib == expectedTripDistrib).all())
              self.assertTrue((indepTripDistrib == expectedIndepTripDistrib).all())

              coOccurTrip = sparse.COO(np.array([[1, 5, 4, 3, 2],
                                                 [0, 2, 3, 2, 2],
                                                 [3, 1, 5, 6, 1]]),
                                          data=np.array([1, 3, 2, 3, 6]))

              coOccurPair = sparse.COO(np.array([[0, 2, 1, 3, 3, 4, 2, 3, 3, 0],
                                                 [3, 3, 2, 1, 3, 3, 4, 5, 6, 5]]),
                                          data=np.array([1, 2, 4, 3, 5, 5, 6, 3, 5, 2]))

              tripDistrib, indepTripDistrib = cd.getDistribForWordI(3, coOccurTrip, coOccurPair)
              self.assertTrue((tripDistrib == expectedTripDistrib).all())
              self.assertTrue((indepTripDistrib == expectedIndepTripDistrib).all())

              coOccurTrip = sparse.COO(np.array([[1, 4, 3, 3],
                                                 [0, 3, 3, 3],
                                                 [3, 5, 6, 3]]),
                                          data=np.array([1, 2, 3, 4]))

              coOccurPair = sparse.COO(np.array([[0, 2, 3, 3, 4, 3, 3],
                                                 [3, 3, 1, 3, 3, 5, 6]]),
                                          data=np.array([1, 2, 3, 4, 5, 3, 6]))

              expectedTripDistrib = np.array([1/10, 4/10, 3/10, 2/10])
              expectedIndepTripDistrib = np.array([1/192, 1/36, 1/24, 5/192])

              tripDistrib, indepTripDistrib = cd.getDistribForWordI(3, coOccurTrip, coOccurPair)
              # print(tripDistrib, expectedTripDistrib)
              # print(indepTripDistrib, expectedIndepTripDistrib)

              self.assertTrue((tripDistrib == expectedTripDistrib).all())
              self.assertTrue((indepTripDistrib == expectedIndepTripDistrib).all())
                            
              return

       def test_calc_distrib_diffs_for_each_word(self):
              coOccurTrip = sparse.COO(np.array([[1, 1, 1, 2],
                                                 [1, 1, 2, 2],
                                                 [2, 3, 3, 3]]),
                                            data=[4, 1, 2, 3])
              coOccurPair = sparse.COO(np.array([[1, 1, 1, 2, 2, 3],
                                                 [1, 2, 3, 2, 3, 3]]),
                                   data=np.array([2, 3, 4, 3, 2, 1]))
              
              expected_trip_distrib_1 = np.array([4/7, 1/7, 2/7])
              expected_trip_distrib_2 = np.array([4/9, 2/9, 3/9])
              expected_trip_distrib_3 = np.array([1/6, 1/3, 1/2])

              expected_ind_distrib_1 = np.array([2*3/81, 2*4/81, 3*4/81])
              expected_ind_distrib_2 = np.array([3*3/64, 3*2/64, 3*2/64])
              expected_ind_distrib_3 = np.array([4*4/49, 4*2/49,2*2/49])


              expected_true_distrib = {1: expected_trip_distrib_1, 2: expected_trip_distrib_2, 3: expected_trip_distrib_3}
              expected_ind_distrib = {1: expected_ind_distrib_1, 2: expected_ind_distrib_2, 3: expected_ind_distrib_3}

              _, indep_distribs, true_distribs = cd.calcDistribDiffsForEachWord(coOccurTrip, coOccurPair)
              print(expected_ind_distrib, expected_true_distrib)
              print(indep_distribs, true_distribs)

              self.assertEqual(len(indep_distribs), len(expected_ind_distrib))
              self.assertEqual(len(true_distribs), len(expected_true_distrib))

              for key, arr in true_distribs.items():
                     assert_array_almost_equal(arr, expected_true_distrib[key])

              for key, arr in indep_distribs.items():
                     assert_array_almost_equal(arr, expected_ind_distrib[key])

       def test_wordcount_arr(self):
              speech_dict_list = {
                     1: ["The the the and of in in", 
                         "the the park of a is",
                         "the and the park a. to to"],
                     2: ["the park in a a to",
                         "and. Park of of of. a",
                         "the the and park park of a park"],
                     3: ["is a of. park Park the",
                         "is to a of of and the",
                         "to a in in in and"],
                     4: ["in in is of The the",
                         "a a To. of and",
                         "in a of Park. And a"]
              }
              
              expected_counts = np.array([[7, 2, 2, 2, 2, 2, 2, 1],
                                          [3, 2, 5, 4, 1, 4, 1, 0],
                                          [2, 2, 2, 3, 3, 3, 2, 2],
                                          [2, 2, 1, 3, 3, 4, 1, 1]])
                                          
              speech_dict_list = {wind: [ps.full_preprocess(speech, filter_stopwords=False, deacc=True) for speech in speech_list] for wind, speech_list in speech_dict_list.items()}
              # print(speech_dict_list)
              input_word_dict = {
                            "the": 0,
                            "and": 1,
                            "park": 2,
                            "of": 3,
                            "in": 4,
                            "a": 5,
                            "to": 6,
                            "is": 7
                            }
              input_word_dict.update({value: key for key, value in input_word_dict.items()})
              
              wordcount_window, word_dict, window_dict = wc.get_wordcount_arr(speech_dict_list, input_word_dict, already_numbers=False)
              # print(wordcount_window)
              self.assertEqual(input_word_dict, word_dict)
              self.assertTrue((wordcount_window == expected_counts).all())
              self.assertEqual(window_dict, {1: 0, 2: 1, 3: 2, 4: 3})

              wordcount_window, word_dict = wc.sort_wordcount_arr(wordcount_window, word_dict)

              expected_word_dict = {
                            "the": 0,
                            "a": 1,
                            "of": 2,
                            "park": 3,
                            "in": 4,
                            "and": 5,
                            "to": 6,
                            "is": 7
                            }
              expected_word_dict.update({value: key for key, value in expected_word_dict.items()})

              expected_counts = np.array([[7, 2, 2, 2, 2, 2, 2, 1],
                                          [3, 4, 4, 5, 1, 2, 1, 0],
                                          [2, 3, 3, 2, 3, 2, 2, 2],
                                          [2, 4, 3, 1, 3, 2, 1, 1]])

              self.assertEqual(expected_word_dict, word_dict)
              self.assertTrue((wordcount_window == expected_counts).all())
       
       def test_indep_estimate_from_two_tuple_arr(self):
              two_tuple_arr = sparse.COO(coords=[[1, 2, 3, 4, 5],
                                                 [5, 3, 1, 2, 3]],
                                          data=[1, 5, 3, 4, 2])
              
              wordcount_for_window = np.array([3, 4, 2, 1, 5, 6])
              expected_indep_probs = [4*6/(21*21), 2*1/(21*21), 1*4/(21*21), 5*2/(21*21), 6*1/(21*21)]

              indep_probs = cd.indep_estimate_from_two_tuple_arr(two_tuple_arr, wordcount_for_window)
              print(indep_probs, expected_indep_probs)

              assert_array_almost_equal(expected_indep_probs, indep_probs)

       def test_make_dict_of_speech_lists(self):
              speeches = pd.DataFrame({"speech": ["The quick brown fox jumps over the lazy dog.", 
                                                    "The city skyline was a breathtaking sight, with tall skyscrapers reaching towards the clouds and twinkling lights illuminating the bustling streets below.",
                                                    "Chair is one of the most essential pieces of furniture for any living room.",
                                                    "Poke bowls are a delicious and healthy meal that are perfect for a quick lunch or dinner.",
                                                    "The Santa Fe Institute is a world-renowned research institution that focuses on complex systems and interdisciplinary approaches to scientific inquiry."],
                                      "wind": [20, 21, 21, 20, 22]
                                      })
              
              expected_speech_dict = {20: [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'], ['poke', 'bowls', 'are', 'a', 'delicious', 'and', 'healthy', 'meal', 'that', 'are', 'perfect', 'for', 'a', 'quick', 'lunch', 'or', 'dinner']], 
                                      21: [['the', 'city', 'skyline', 'was', 'a', 'breathtaking', 'sight', 'with', 'tall', 'skyscrapers', 'reaching', 'towards', 'the', 'clouds', 'and', 'twinkling', 'lights', 'illuminating', 'the', 'bustling', 'streets', 'below'], ['chair', 'is', 'one', 'of', 'the', 'most', 'essential', 'pieces', 'of', 'furniture', 'for', 'any', 'living', 'room']],
                                      22: [['the', 'santa', 'fe', 'institute', 'is', 'a', 'world', 'renowned', 'research', 'institution', 'that', 'focuses', 'on', 'complex', 'systems', 'and', 'interdisciplinary', 'approaches', 'to', 'scientific', 'inquiry']]}
              
              speech_dict = ps.make_dict_of_speech_lists(speeches, filter_stopwords=False)
       
              self.assertEqual(speech_dict, expected_speech_dict)

              tiny_speech_dict = {20: [["hi", "bye", "morning"], ["morning", "me"]],
                                  21: [["desk", "chair", "morning"], ["hi", "desk", "bye", "hi"], ["me", "david", "me", "david", "me"]],
                                  22: [["david", "fear"], ["morning", "chair"]]}
              word_dict = { 
                     "bye": 0,
                     "morning": 1,
                     "hi": 4,
                     "me": 2,
                     "desk": 3,
                     "chair": 5,
                     "david": 6,
                     "fear": 7
              }
              word_dict.update({val: key for key, val in word_dict.items()})
              window_dict = {22: 1, 21: 2, 20: 0}
              expected_int_dict = {0: [np.array([4, 0, 1], dtype=np.uint16), np.array([1, 2], dtype=np.uint16)],
                                   2: [np.array([3, 5, 1], dtype=np.uint16), np.array([4, 3, 0, 4], dtype=np.uint16), np.array([2, 6, 2, 6, 2], dtype=np.uint16)],
                                   1: [np.array([6, 7], dtype=np.uint16), np.array([1, 5], dtype=np.uint16)]}
              
              speech_dict_tiny_int = ps.convert_words_to_ints(tiny_speech_dict, word_dict, window_dict)
              # print(speech_dict_tiny_int)
              for wind, speech_list in expected_int_dict.items():
                     self.assertIn(wind, speech_dict_tiny_int)
                     for i, speech in enumerate(speech_list):
                            self.assertTrue((speech == speech_dict_tiny_int[wind][i]).all())
       
       def test_n_wise_cooccurence_distribs(self):
              speech_list_dict = {0: [np.array([0, 3, 4, 5, 3, 2], dtype=np.uint16), 
                                      np.array([4, 1, 3, 6, 7, 3], dtype=np.uint16), 
                                      np.array([2, 4, 6, 1, 3, 1], dtype=np.uint16)],
                                   1: [np.array([1, 3, 2, 1, 2], dtype=np.uint16), 
                                       np.array([2, 1, 3, 2, 3], dtype=np.uint16)]}
              ld.dump_one_by_one(speech_list_dict, "testpair_speech_dict", parent="objects/testpair")

              expected_wind_cooccur = {
                     1: {
                            (1, 1, 2): 1,
                            (1, 1, 3): 1,
                            (1, 2, 2): 2,
                            (1, 2, 3): 6,
                            (1, 3, 3): 1,
                            (2, 2, 3): 2,
                            (2, 3, 3): 1
                            
                     },
                     0: {
                            (0, 3, 4): 1,
                            (0, 3, 5): 1,
                            (0, 4, 5): 1,
                            (1, 1, 3): 1,
                            (1, 1, 6): 1,
                            (1, 2, 4): 1,
                            (1, 2, 6): 1,
                            (1, 3, 4): 2,
                            (1, 3, 6): 3,
                            (1, 3, 7): 1,
                            (1, 4, 6): 2,
                            (1, 6, 7): 1,
                            (2, 3, 4): 1,
                            (2, 3, 5): 1,
                            (2, 4, 5): 1,
                            (2, 4, 6): 1,
                            (3, 3, 4): 1,
                            (3, 3, 5): 1,
                            (3, 3, 6): 1,
                            (3, 3, 7): 1,
                            (3, 4, 5): 2,
                            (3, 4, 6): 2,
                            (3, 6, 7): 2
                     }
              }

              expected_tot_cooccur = {
                     (0, 3, 4): 1,
                     (0, 3, 5): 1,
                     (0, 4, 5): 1,
                     (1, 1, 2): 1,
                     (1, 1, 3): 2,
                     (1, 1, 6): 1,
                     (1, 2, 2): 2,
                     (1, 2, 3): 6,
                     (1, 2, 4): 1,
                     (1, 2, 6): 1,
                     (1, 3, 3): 1,
                     (1, 3, 4): 2,
                     (1, 3, 6): 3,
                     (1, 3, 7): 1,
                     (1, 4, 6): 2,
                     (1, 6, 7): 1,
                     (2, 2, 3): 2,
                     (2, 3, 3): 1,
                     (2, 3, 4): 1,
                     (2, 3, 5): 1,
                     (2, 4, 5): 1,
                     (2, 4, 6): 1,
                     (3, 3, 4): 1,
                     (3, 3, 5): 1,
                     (3, 3, 6): 1,
                     (3, 3, 7): 1,
                     (3, 4, 5): 2,
                     (3, 4, 6): 2,
                     (3, 6, 7): 2
              }

              winds = ld.get_keys_for_name(os.path.join("objects", "testpair"), f"testpair_speech_dict", int)
              print(winds)
              wind_cooccur_arr, wind_cooccur_dct, total_cooccur_dct = cd.genNWiseCooccurenceDistribs("testpair", winds, 3, frame=4, num_focal_words=500, num_context_words=1000, normalize=False, parallel=True, par_memo=False, tuple_dict=None, tuple_thresh=1, return_dict=True)
              self.assertEqual(wind_cooccur_dct, expected_wind_cooccur)
              self.assertEqual(total_cooccur_dct, expected_tot_cooccur)

              for wind, dct in wind_cooccur_dct.items():
                     for tup, val in dct.items():
                            self.assertEqual(wind_cooccur_arr[wind][tup], val)
              
              tuple_dict = {(1, 2, 3): 2, (3, 4, 5): 2, (2, 3, 4): 2, (4, 5, 6): 2, (1, 1, 2): 0, (1, 2, 2): 0, (1, 3, 6): 0}
              wind_cooccur_arr, wind_cooccur_dct, total_cooccur_dct = cd.genNWiseCooccurenceDistribs("testpair", winds, 3, frame=4, num_focal_words=500, num_context_words=1000, normalize=False, parallel=True, par_memo=False, tuple_dict=tuple_dict, tuple_thresh=2, return_dict=True)
              # print(wind_cooccur_dct)
              # print(total_cooccur_dct)

              expected_wind_cooccur = {
                     1: {
                            (1, 2, 3): 6,
                            (2, 3, 4): 0,
                            (3, 4, 5): 0,
                            (4, 5, 6): 0
                     },
                     0: {
                            (1, 2, 3): 0,
                            (2, 3, 4): 1,
                            (3, 4, 5): 2,
                            (4, 5, 6): 0
                     }
              }
              expected_tot_cooccur = {
                     (1, 2, 3): 6,
                     (2, 3, 4): 1,
                     (3, 4, 5): 2,
                     (4, 5, 6): 0
              }

              self.assertEqual(wind_cooccur_dct, expected_wind_cooccur)
              self.assertEqual(total_cooccur_dct, expected_tot_cooccur)

              for wind, dct in wind_cooccur_dct.items():
                     for tup, val in dct.items():
                            self.assertEqual(wind_cooccur_arr[wind][tup], val)
              
              cd.genNWiseCooccurenceDistribs("testpair", winds, 3, frame=4, num_focal_words=500, num_context_words=1000, normalize=False, parallel=True, par_memo=False, tuple_dict=None, tuple_thresh=1)
              keys = ld.get_keys_for_name("objects/tinypair", "tinypair_3tuple_dict", int)

              for key in keys:
                     print(key)
                     with open(f"objects/tinypair/tinypair_3tuple_dict_{key}", "rb") as f:
                            dct = pkl.load(f)
                     
                     print(dct)
                     self.assertEqual(dct, expected_wind_cooccur[key])

       
       def test_dump(self):
              sample_speeches, sample_speech_dict, sample_window_wordcount, sample_word_dict, sample_window_dict = ps.generate_sample_data(manyWindows=True)
              ld.dump_one_by_one(sample_speech_dict, "testdump_speech_dict", parent="objects/testdump")
              ld.dump_counts(sample_window_wordcount, sample_word_dict, sample_window_dict, "testdump", parent="objects/testdump")

              winds = ld.get_keys_for_name("objects/testdump", "testdump_speech_dict", int)

              for wind in winds:
                     speech_list = ld.read_by_key("testdump_speech_dict", wind, "objects/testdump")
                     self.assertEqual(speech_list, sample_speech_dict[wind])

              for wind in sample_speech_dict:
                     self.assertIn(wind, winds)
              
              window_wordcount, word_dict, window_dict = ld.load_counts("testdump")
              self.assertTrue((window_wordcount == sample_window_wordcount).all())
              self.assertEqual(word_dict, sample_word_dict)
              self.assertEqual(window_dict, sample_window_dict)

       def test_pair_trip_dict(self):
              speech_list_dict = {0: [np.array([1, 2, 2, 3], dtype=np.uint16), 
                                      np.array([3, 3, 1, 2, 2], dtype=np.uint16),
                                      np.array([1, 2, 4], dtype=np.uint16)],
                                   1: [np.array([1, 3, 2, 1, 2], dtype=np.uint16), 
                                       np.array([2, 1, 3, 2, 3], dtype=np.uint16)]}

              ld.dump_one_by_one(speech_list_dict, "testsample_speech_dict", parent="objects/testsample")
              
              expected_trip = {
                     (1, 1, 2): 1,
                     (1, 1, 3): 1,
                     (1, 2, 2): 4,
                     (1, 2, 3): 11,
                     (1, 3, 3): 2,
                     (2, 2, 3): 4,
                     (2, 3, 3): 2,
                     (1, 2, 4): 1
              }
              expected_pair = {
                     (1, 1): 1,
                     (1, 2): 10,
                     (1, 3): 7,
                     (2, 2): 4,
                     (2, 3): 10,
                     (3, 3): 2,
                     (1, 4): 1,
                     (2, 4): 1
              }

              pair_dict, trip_dict = cd.gen_pair_trip_dict("testsample", [0, 1], 1, frame=4, num_focal_words=500, num_context_words=1000, normalize=False, parallel=True, par_memo=False)
              print(trip_dict)
              print(pair_dict)
              self.assertEqual(pair_dict, expected_pair)
              self.assertEqual(trip_dict, expected_trip)

              for i in range(25):
                     pair_dict, trip_dict = cd.gen_pair_trip_dict("testsample", [0, 1], 0.5, frame=4, num_focal_words=500, num_context_words=1000, normalize=False, parallel=True, par_memo=False)
                     for trip in trip_dict:
                            candidate_tuples = [(trip[0], trip[1]), (trip[1], trip[2]), (trip[0], trip[2])]
                            for tup in candidate_tuples:
                                   self.assertIn(tup, pair_dict)

       def test_group_windows(self):
              winds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              self.assertEqual([[1, 2, 3],[4, 5, 6],[7, 8, 9]], cd.group_windows(winds, 3))
              self.assertEqual([[1, 2, 3, 4], [5, 6, 7, 8]], cd.group_windows(winds, 4))
              self.assertEqual([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], cd.group_windows(winds, 5))
              self.assertEqual([[1, 2, 3, 4, 5, 6]], cd.group_windows(winds, 6))
              self.assertEqual([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], cd.group_windows(winds, 2))
              self.assertEqual([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], cd.group_windows(winds, 1))

       def test_word_window_context_distribs(self):
              speech_dict = {20: [np.array([0,1,0,3,2,2,3,4], dtype=np.uint16), np.array([3,3,4,0,3], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([5, 5], dtype=np.uint16)], 
                             21: [np.array([1,2,0,2,2], dtype=np.uint16), np.array([4,1,0], dtype=np.uint16), np.array([0], dtype=np.uint16), np.array([5, 5], dtype=np.uint16)]}
              
              wordcount_window, word_dict, window_dict = wc.get_wordcount_arr(speech_dict, already_numbers=True)
              ld.dump_counts(wordcount_window, word_dict, window_dict, "testcontext", os.path.join("objects", "testcontext"))
             
              expected_counts = np.array(
                     [[[2, 2, 1, 3, 1],
                       [2, 0, 0, 1, 0],
                       [1, 0, 2, 4, 1],
                       [3, 1, 4, 2, 4]],
                      [[0, 2, 3, 0, 1],
                       [2, 0, 1, 0, 1],
                       [3, 1, 4, 0, 0],
                       [0, 0, 0, 0, 0]]]
              )

              ld.dump_one_by_one(speech_dict, "testcontext_speech_dict", os.path.join("objects", "testcontext"), window_dict=window_dict)
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 2, start_ind=0, end_ind=4, top_m=5, sample_equally=False)

              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array(
                     [[[0, 0, 0, 0, 0],
                       [2, 0, 0, 1, 0],
                       [1, 0, 2, 4, 1],
                       [3, 1, 4, 2, 4]],
                      [[0, 0, 0, 0, 0],
                       [2, 0, 1, 0, 1],
                       [3, 1, 4, 0, 0],
                       [0, 0, 0, 0, 0]]]
              )
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 2, start_ind=1, end_ind=4, top_m=5, sample_equally=False)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array(
                     [[[0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 2],
                       [3, 1, 4],
                       [1, 0, 1]],
                      [[0, 0, 0],
                       [0, 0, 0],
                       [3, 1, 4],
                       [0, 0, 0],
                       [1, 1, 0]]]
              )

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 2, start_ind=2, end_ind=5, top_m=3, sample_equally=False)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array(
                     [[[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 2, 4, 1],
                       [0, 0, 4, 2, 4],
                       [0, 0, 1, 4, 0]],
                      [[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 4, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]]
              )
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 2, start_ind=2, end_ind=5, top_m=3, sample_equally=False, symmetric=True)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array(
                     [[[2, 2, 1, 3, 1],
                       [0, 0, 0, 0, 0],
                       [1, 0, 2, 4, 1]],
                      [[0, 2, 3, 0, 1],
                       [0, 0, 0, 0, 0],
                       [3, 1, 4, 0, 0]]]
              )
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 2, start_ind=0, end_ind=5, top_m=5, sample_equally=True, min_count=2, symmetric=False)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array(
                     [[[2, 0, 1],
                       [0, 0, 0],
                       [1, 0, 2]],
                      [[0, 0, 3],
                       [0, 0, 0],
                       [3, 0, 4]]]
              )

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 2, start_ind=0, end_ind=5, top_m=5, sample_equally=True, min_count=2, symmetric=True)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array(
                     [[[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 2]],
                      [[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 2]]]
              )

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 2, start_ind=5, end_ind=6, top_m=6, sample_equally=False, symmetric=False)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array(
                     [[[2, 2, 1, 3, 1, 0],
                       [0, 0, 0, 0, 0, 0],
                       [1, 0, 2, 4, 1, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 2]],
                      [[0, 2, 3, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0],
                       [3, 1, 4, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 2]]]
              )

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 2, start_ind=0, end_ind=6, top_m=6, sample_equally=True, min_count=1, symmetric=False)
              self.assertEqual(context_counts.shape, expected_counts.shape)
              self.assertTrue((context_counts[:,5,:] == expected_counts[:,5,:]).all())
              self.assertTrue((context_counts <= expected_counts).all())


              speech_dict = {20: [np.array([0, 2, 2, 5], dtype=np.uint16), np.array([0, 5], dtype=np.uint16), np.array([0, 2, 0, 5], dtype=np.uint16), np.array([1, 3, 1, 3], dtype=np.uint16)], 
                             21: [np.array([2, 2, 5, 0], dtype=np.uint16), np.array([0, 2, 5, 0], dtype=np.uint16), np.array([0, 5], dtype=np.uint16), np.array([1, 1, 3, 3], dtype=np.uint16)]}
              
              wordcount_window, word_dict, window_dict = wc.get_wordcount_arr(speech_dict, already_numbers=True)
              ld.dump_counts(wordcount_window, word_dict, window_dict, "testcontext", os.path.join("objects", "testcontext"))
             
              expected_counts = np.array(
                     [[[0, 0, 3, 0, 0, 2],
                       [0, 0, 0, 3, 0, 0],
                       [3, 0, 2, 0, 0, 1],
                       [0, 3, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [2, 0, 1, 0, 0, 0]],
                      [[0, 0, 1, 0, 0, 3],
                       [0, 2, 0, 1, 0, 0],
                       [1, 0, 2, 0, 0, 2],
                       [0, 1, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [3, 0, 2, 0, 0, 0]]]
              )
              expected_pcts = expected_counts / expected_counts.sum(axis=1, keepdims=True)

              ld.dump_one_by_one(speech_dict, "testcontext_speech_dict", os.path.join("objects", "testcontext"), window_dict=window_dict)
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 1, start_ind=0, end_ind=6, top_m=6, sample_equally=False, symmetric=True)
              self.assertTrue((context_counts == expected_counts).all())
              self.assertTrue((np.abs(np.nan_to_num(context_counts / context_counts.sum(axis=2, keepdims=True), nan=0) - np.nan_to_num(context_pcts, nan=0)) < 0.000001).all())

              expected_counts = np.array(
                     [[[0, 0, 3, 0, 0, 2],
                       [0, 0, 0, 0, 0, 0],
                       [3, 0, 2, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [2, 0, 1, 0, 0, 0]],
                      [[0, 0, 1, 0, 0, 3],
                       [0, 0, 0, 0, 0, 0],
                       [1, 0, 2, 0, 0, 2],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [3, 0, 2, 0, 0, 0]]]
              )

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 1, start_ind=0, end_ind=6, top_m=6, sample_equally=True, symmetric=True, min_count=2)
              print(context_counts)
              self.assertTrue((context_counts <= expected_counts).all())
              self.assertTrue((context_counts[:,0,:] <= expected_counts[:,0,:]).all())
              self.assertTrue((context_counts[:,2,:] <= expected_counts[:,2,:]).all())
              assert_array_almost_equal(context_counts / context_counts.sum(axis=-1, keepdims=True), context_pcts)
              
              speech_dict = {20: [np.array([0, 0, 0, 3, 4, 3, 4, 3, 2], dtype=np.uint16), np.array([2, 4, 2, 4, 3, 3])], 
                             21: [np.array([3, 3, 3, 4, 4, 4], dtype=np.uint16), np.array([0, 0, 0, 3, 3, 3, 2, 4, 4], dtype=np.uint16)]}
              
              wordcount_window, word_dict, window_dict = wc.get_wordcount_arr(speech_dict, already_numbers=True)
              ld.dump_counts(wordcount_window, word_dict, window_dict, "testcontext", os.path.join("objects", "testcontext"))
              ld.dump_one_by_one(speech_dict, "testcontext_speech_dict", os.path.join("objects", "testcontext"), window_dict=window_dict)

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext", 1, start_ind=1, end_ind=5, top_m=6, sample_equally=True, min_count=1)
              print(foc_dict)
              print(context_counts)
              self.assertEqual(context_counts.shape, (2, 5, 6))
              self.assertTrue((context_counts[:,2,:] == 0).all())
              self.assertTrue((context_counts[:,[0, 1],:] == 0).all())
              print(context_counts) 

              speech_dict = ps.generate_random_speeches(4, [6/20, 5/20, 4/20, 3/20, 2/20], 1000, 200)
              wordcount_window, word_dict, window_dict = wc.get_wordcount_arr(speech_dict, already_numbers=True)
              ld.dump_counts(wordcount_window, word_dict, window_dict, "testcontext2", os.path.join("objects", "testcontext2"))
              ld.dump_one_by_one(speech_dict, "testcontext2_speech_dict", os.path.join("objects", "testcontext2"), window_dict=window_dict)

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext2", 5, start_ind=0, end_ind=5, top_m=5, sample_equally=True, min_count=35000, common_sample=10000)

              expected_totals = np.ones((4, 3))*100000
              totals = context_counts.sum(axis=-1)

              print(totals)
              self.assertTrue((np.abs(expected_totals - totals) < 7500).all())

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext2", 5, start_ind=0, end_ind=5, top_m=5, sample_equally=True, min_count=70000, group_length=2)
              expected_totals = np.ones((2, 3))*800000
              totals = context_counts.sum(axis=-1)

              print(totals)
              self.assertTrue((np.abs(expected_totals - totals) < 8000).all())

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext2", 5, start_ind=0, end_ind=5, top_m=5, sample_equally=True, min_count=35000)
              expected_totals = np.ones((4, 3))*400000
              totals = context_counts.sum(axis=-1)

              print(totals)
              self.assertTrue((np.abs(expected_totals - totals) < 7500).all())

              
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext2", 5, start_ind=0, end_ind=5, top_m=5, sample_equally=False, min_count=35000)
              expected_totals = np.ones((4, 5))*np.array([[600000, 500000, 400000, 300000, 200000]])
              print(expected_totals)
              totals = context_counts.sum(axis=-1)

              print(totals)
              self.assertTrue((np.abs(expected_totals - totals) < 10000).all())
              
              speech_dict = {20: [np.array([0,1,0,3,2,2,3,4], dtype=np.uint16), np.array([3,3,4,0,3], dtype=np.uint16), np.array([2], dtype=np.uint16), np.array([5, 5], dtype=np.uint16)], 
                             21: [np.array([1,2,0,2,2], dtype=np.uint16), np.array([4,1,0], dtype=np.uint16), np.array([0], dtype=np.uint16), np.array([5, 5], dtype=np.uint16)],
                             22: [np.array([1, 2], dtype=np.uint16), np.array([3, 4], dtype=np.uint16)],
                             23: [np.array([1, 3], dtype=np.uint16), np.array([2, 4], dtype=np.uint16)],
                             24: [np.array([1, 4], dtype=np.uint16), np.array([2, 3], dtype=np.uint16)]}
              
              wordcount_window, word_dict, window_dict = wc.get_wordcount_arr(speech_dict, already_numbers=True)
              ld.dump_counts(wordcount_window, word_dict, window_dict, "testcontext3", os.path.join("objects", "testcontext3"))
              ld.dump_one_by_one(speech_dict, "testcontext3_speech_dict", os.path.join("objects", "testcontext3"), window_dict=window_dict)

              expected_counts = np.array(
                     [[[2, 2, 1, 3, 1],
                       [2, 0, 0, 1, 0],
                       [1, 0, 2, 4, 1],
                       [3, 1, 4, 2, 4],
                       [1, 0, 1, 4, 0]],
                      [[0, 2, 3, 0, 1],
                       [2, 0, 1, 0, 1],
                       [3, 1, 4, 0, 0],
                       [0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0]],
                      [[0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0]],
                      [[0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0]],
                      [[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0]]]
              )
              
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext3", 2, start_ind=0, end_ind=5, top_m=5)
              print(context_counts)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array([[[2, 4, 4, 3, 2],
                                           [4, 0, 1, 1, 1],
                                           [4, 1, 6, 4, 1],
                                           [3, 1, 4, 2, 4],
                                           [2, 1, 1, 4, 0]],
                                          [[0, 0, 0, 0, 0],
                                           [0, 0, 1, 1, 0],
                                           [0, 1, 0, 0, 1],
                                           [0, 1, 0, 0, 1],
                                           [0, 0, 1, 1, 0]]])

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext3", 2, start_ind=0, end_ind=5, top_m=5, group_length=2)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array([[[2, 4, 4, 3, 2],
                                           [4, 0, 2, 1, 1],
                                           [4, 2, 6, 4, 1],
                                           [3, 1, 4, 2, 5],
                                           [2, 1, 1, 5, 0]]])
              
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext3", 2, start_ind=0, end_ind=5, top_m=5, group_length=3)
              print(expected_counts)
              self.assertTrue((context_counts == expected_counts).all())

              expected_counts = np.array([[[2, 4, 4, 3, 2],
                                           [4, 0, 2, 2, 2],
                                           [4, 2, 6, 5, 2],
                                           [3, 2, 5, 2, 5],
                                           [2, 2, 2, 5, 0]]])
              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext3", 2, start_ind=0, end_ind=5, top_m=5, group_length=5)
              self.assertTrue((context_counts == expected_counts).all())

              speech_dict = {20: [np.array([0, 0, 0], dtype=np.uint16), np.array([1, 1, 1], dtype=np.uint16), np.array([3, 3], dtype=np.uint16), np.array([4, 4], dtype=np.uint16),], 
                             21: [np.array([0, 0, 1], dtype=np.uint16), np.array([2, 2, 2], dtype=np.uint16), np.array([3], dtype=np.uint16)],
                             22: [np.array([1, 1, 1], dtype=np.uint16), np.array([2, 1, 1], dtype=np.uint16), np.array([3, 3], dtype=np.uint16), np.array([4, 4, 4], dtype=np.uint16)],
                             23: [np.array([1, 1, 2], dtype=np.uint16), np.array([0, 0, 0], dtype=np.uint16), np.array([3], dtype=np.uint16)]}
              
              wordcount_window, word_dict, window_dict = wc.get_wordcount_arr(speech_dict, already_numbers=True)
              ld.dump_counts(wordcount_window, word_dict, window_dict, "testcontext4", os.path.join("objects", "testcontext4"))
              ld.dump_one_by_one(speech_dict, "testcontext4_speech_dict", os.path.join("objects", "testcontext4"), window_dict=window_dict)

              context_pcts, context_pctvar, context_counts, context_variances, foc_dict = cd.genWordWindowContextDistribsStartToEnd("testcontext4", 1, start_ind=0, end_ind=4, top_m=5, group_length=2, sample_equally=True, min_count=2)
              print(context_counts)
              self.assertEqual(context_counts.shape, (2, 4, 5))



       def test_compile_list_of_changes(self):
              contexts = np.array([[[0.5, 0.1, 0.4],
                                    [0.3, 0.4, 0.3]],
                                   [[0.2, 0.3, 0.5],
                                    [0.4, 0.4, 0.2]],
                                   [[0.6, 0.3, 0.1],
                                    [0.3, 0.5, 0.2]]])
              expected_change_summands = np.array(
                     [[[0.3, 0.2, 0.1],
                       [0.1, 0.0, 0.1]],
                      [[0.4, 0.0, 0.4],
                       [0.1, 0.1, 0.0]]]
              )
              expected_changes = np.array(
                     [[0.6, 0.2],
                      [0.8, 0.2]]
              )
              changes, change_summands = cd.compile_list_of_changes(contexts, 2, comp_func=dist.make_shift_row_arrs)

              # print("change summands", change_summands)
              assert_array_almost_equal(expected_change_summands, change_summands, decimal=8)
              assert_array_almost_equal(changes, expected_changes, decimal=8)
              
              contexts = np.array([[[0.5, 0.1, 0.4],
                                    [0.3, 0.4, 0.3]],
                                   [[0.2, 0.3, 0.5],
                                    [0.4, 0.4, 0.2]],
                                   [[0.5, 0.1, 0.8],
                                    [0.4, 0.3, 0.8]]])
              
              expected_change_summands = np.array(
                     [[[0.0, 0.0, 0.4],
                       [0.1, 0.1, 0.5]],
                      [[0.3, 0.2, 0.3],
                       [0.0, 0.1, 0.6]]]
              )
              expected_changes = np.array(
                     [[0.4, 0.7],
                      [0.8, 0.7]]
              )
              changes, change_summands = cd.compile_list_of_changes(contexts, 2, comp_func=dist.make_past_present_arrs)

              print("change summands", change_summands)
              assert_array_almost_equal(expected_change_summands, change_summands, decimal=8)
              assert_array_almost_equal(changes, expected_changes, decimal=8)
       
       def test_calc_projections(self):
              val_arr = np.array([[[1, 2, 3],[0, -1, 3],[4, 5, 6]],
                                   [[2, 1, 4],[2, -3, 4],[5, 6, 7]],
                                   [[8, 9, 7],[-2, 5, 9],[4, 5, 6]],
                                   [[5, 6, 7],[-5, 6, 6],[3, 9, 4]]])
              before_arr, after_arr = dist.make_shift_row_arrs(val_arr)
              
              expected_diffs = np.array([[[1, -1, 1],[2, -2, 1],[1, 1, 1]],
                                          [[6, 8, 3],[-4, 8, 5],[-1, -1, -1]],
                                          [[-3, -3, 0],[-3, 1, -3],[-1, 4, -2]]])
              dot_products = np.array([[[6, -8, 3], [-8, -16, 5], [-1, -1, -1]],
                                       [[-18, -24, 0],[12, 8, -15],[1, -4, 2]]])
              expected_projection_mags = np.array([[1, -19, -3],[-42, 5, -1]])
              projection_mags = dist.calc_projections(before_arr, after_arr)

              self.assertTrue((projection_mags == expected_projection_mags).all())
       
       def test_sample_contexts_wind(self):
              speech_dict = {0: [np.array([0,1,0,3], dtype=np.uint16), np.array([3,3,2], dtype=np.uint16), np.array([2, 2, 3, 0, 1, 2], dtype=np.uint16), np.array([1, 2, 3, 2, 1], dtype=np.uint16)]}
              
              ld.dump_one_by_one(speech_dict, "testsamplecontexts_speech_dict", os.path.join("objects", "testsamplecontexts"))

              possible_contexts = np.array([[0, 1, 0],
                                            [1, 0, 3],
                                            [3, 3, 2],
                                            [2, 2, 3],
                                            [2, 3, 0],
                                            [3, 0, 1],
                                            [0, 1, 2],
                                            [1, 2, 3],
                                            [2, 3, 2],
                                            [3, 2, 1]])
              
              contexts = cw.sample_contexts_wind("testsamplecontexts", 0, 3, samples=18000)
              contexts = contexts.reshape(-1, 1, 3)
              possible_contexts = possible_contexts.reshape(1, -1, 3)
              print(contexts.shape)
              print(possible_contexts.shape)
              context_counts = (contexts == possible_contexts).all(axis=-1)
              self.assertEqual(context_counts.sum(), 18000)
              self.assertTrue((context_counts.sum(axis=1) == np.ones((18000,))).all())

              contexts_0 = np.array([[[0, 1, 0],
                                      [1, 0, 3]]])
              contexts_1 = np.array([[[3, 3, 2]]])
              contexts_2 = np.array([[[2, 2, 3],
                                      [2, 3, 0],
                                      [3, 0, 1],
                                      [0, 1, 2]]])
              contexts_3 = np.array([[[1, 2, 3],
                                      [2, 3, 2],
                                      [3, 2, 1]]])
              all_poss_contexts = [contexts_0, contexts_1, contexts_2, contexts_3]
              lengths = [4, 3, 6, 5]

              for poss_context, length in zip(all_poss_contexts, lengths):
                     print(poss_context, length)
                     context_count = (contexts == poss_context).all(axis=-1).sum(axis=0)
                     overall_count = context_count.sum()
                     expectation = length * 1000
                     self.assertLess(overall_count, expectation + 200) # PROB
                     self.assertGreater(overall_count, expectation - 200) # PROB
                     print(poss_context, length, context_count, expectation, overall_count)

                     num_contexts = poss_context.shape[1]

                     self.assertTrue((context_count < (overall_count / num_contexts) + 100).all())
                     self.assertTrue((context_count > (overall_count / num_contexts) - 100).all())
       
       def test_context_to_count(self):
              contexts = np.array([[4, 3, 4],
                                   [0, 0, 3],
                                   [1, 1, 3],
                                   [1, 4, 5],
                                   [5, 3, 3],
                                   [2, 1, 0],
                                   [3, 2, 3],
                                   [3, 1, 4],
                                   [4, 2, 5],
                                   [3, 1, 0]])
              
              expected_counts = np.array([[0, 0, 0, 1, 2],
                                          [2, 0, 0, 1, 0],
                                          [0, 2, 0, 1, 0],
                                          [0, 1, 0, 0, 2],
                                          [0, 0, 0, 2, 1],
                                          [1, 1, 1, 0, 0],
                                          [0, 0, 1, 2, 0],
                                          [0, 1, 0, 1, 1],
                                          [0, 0, 1, 0, 2],
                                          [1, 1, 0, 1, 0]])
              
              counts = cw.context_to_count(contexts, 4)
              print(counts)

              self.assertTrue((counts == expected_counts).all()) 
              self.assertEqual(counts.sum(), 30)

              contexts = np.random.choice([0, 1, 2, 3, 4], size=(1000, 10))
              counts = cw.context_to_count(contexts, 3)
              self.assertEqual(counts.sum(), 10000)
              self.assertEqual(counts.shape, (1000, 4))
              self.assertLess(counts.sum(axis=0)[-1], 4000 + 300)
              self.assertGreater(counts.sum(axis=0)[-1], 4000 - 300)

if __name__ == "__main__":
    # unittest.main()
    tcd = TestContextDistribs()
    tcd.test_word_window_context_distribs()
