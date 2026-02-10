# Variables - change these to match your system
HOST_BIDS="/Users/peerherholz/google_drive/GitHub/ffrprep/ffrprep_raw_data"
HOST_OUT="/Users/peerherholz/google_drive/GitHub/ffrprep/ffrprep_raw_data/derivatives"
IMAGE="ffrprep:local"   # or sparkcsd/ffrprep:local if you built locally

mkdir -p "$HOST_OUT"

docker run --rm -it \
  -v "$HOST_BIDS":/data \
  -v "$HOST_OUT":/out \
  --user "$(id -u):$(id -g)" \
  "$IMAGE" \
  /data /out participant \
  --participant_label 21 \
  --stage preprocessing \
  --ref_channels M1 M2 \
  --l_freq 65 \
  --h_freq 2000 \
  --run 2 \
  --task active \
  --n_procs 1 \
  --reject-eeg 0.000075 \
  --save-each-node \
  --picks Cz \
  --tmin -0.04 \
  --tmax 0.4 \
  --baseline -0.04 0 \
  --on-missing warn