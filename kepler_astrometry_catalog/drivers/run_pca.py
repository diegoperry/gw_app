from kepler_astrometry_catalog.cotrending import run_pca, calc_pca_residual

quarter = 12
sampleid = 'brightestnonsat100_rot'

run_pca(quarter=quarter,dvacorr=1, sampleid = sampleid)
calc_pca_residual(quarter=quarter, sampleid=sampleid)

# run_pca(quarter=quarter, dvacorr=1, useTrunSVD=1)
# calc_pca_residual(quarter = quarter, useTrunSVD=1)
