SEED         = 4321   # seed to fix rand.
PGD_Linf_EPS = 4/255. # L_inf epsilon for PGD
BATCH_SIZE   = 128    # Batch size (for training, attacks, and defeses)
RS_N0        = 10     # n0 samples for prediction using the smoothed model
RS_N         = 1000   # n samples for certification using the smoothed model
RS_ALPHA     = 0.05   # for obtaining >=1-alpha confidence in the certificate
