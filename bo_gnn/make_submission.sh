mkdir MixedInspiringLamePuns
cp -r config/agents MixedInspiringLamePuns
cp config/init.sh MixedInspiringLamePuns
cp config/conda.yaml MixedInspiringLamePuns
cp -r config/param_configurations MixedInspiringLamePuns
cp -r config/trained_model_checkpoints MixedInspiringLamePuns
cp -r config/models MixedInspiringLamePuns
cp config/train_gnn.py MixedInspiringLamePuns
cp -r config/data_utils MixedInspiringLamePuns
mkdir MixedInspiringLamePuns/data
cp -r config/data/3_anonymous/ MixedInspiringLamePuns/data
zip -r config/MixedInspiringLamePuns.zip MixedInspiringLamePuns
mv MixedInspiringLamePuns ../submissions