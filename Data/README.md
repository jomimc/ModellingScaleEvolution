## Directory Structure
* Melodies
  * "folk_corpora_countries.txt" :: Countries where melodic corpora come from
  * "folk_corpora_groups.txt"   :: Regional groups of melodic corpora
  * "folk_corpora_scale_degree_counts.npy"  :: Distributions of the number of scale degrees per corpus
  * "melodic_database_summary.csv"  ::  Summary of melodic corpora. In this work we only study those labelled "Folk" under the column "ctype"
  * "melodic_interval_dist_per_corpus.npy"  ::  Distributions of absolute melodic intervals (from 0 to 14 semitones) for each corpus
  * "melodic_interval_scaling_coef_per_corpus.npy"  ::  Scaling coefficients for melodic interval distributions
* ProductionPerception
  * Contains data extracted from papers using g3data
  * "sdt_fit.py" contains code that is used to extract sigma from various sources
* Scales
  * "all_scales_data.pkl"   ::  All scales used in this work, 
  * "dataset_damusc.pkl"    ::  Scales taken from the Database of Musical Scales (DaMuSc)
  * "dataset_garland.pkl"   ::  Scales taken from the Garland Encylopedia of Music collection
  * "dataset_steven.pkl"    ::  Scales taken from Brown et al.,  Nature Humanities and Social Science Communications (2025)
  * "step_sizes_kde.npy" ::  A list of values of step sizes, taken from Vocal scales
  * "step_sizes.npy"    :: A kernel-density estimate created using the list of Vocal step sizes, used for sampling new scales


Some data is stored in numpy ".npy" files. They can be read using python code:
> numpy.load(filename)

Some data is stored as pandas dataframes in pickle files, ".pkl". They can be read using python code:
> pandas.read_pickle(filename)


