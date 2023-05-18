import context_distribs as cd


if __name__ == "__main__":
    (context_pcts, context_pctvar, 
    context_counts, context_variances, 
    foc_dict) = cd.genWordWindowContextDistribsStartToEnd("full", frame=10, start_ind=0, end_ind=3000, 
                                                        top_m=10000, sample_equally=True, min_count=3000)

