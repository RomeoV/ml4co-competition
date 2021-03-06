mkdir MixedInspiringLamePuns
cp -r config/agents MixedInspiringLamePuns
cp config/init.sh MixedInspiringLamePuns
cp config/conda.yaml MixedInspiringLamePuns
cp -r config/param_configurations MixedInspiringLamePuns
cp -r config/heuristics_schedules MixedInspiringLamePuns
cp -r config/trained_model_checkpoints MixedInspiringLamePuns
cp -r config/models MixedInspiringLamePuns
cp -r config/schedules MixedInspiringLamePuns
cp config/train_gnn.py MixedInspiringLamePuns
cp config/models/callbacks.py MixedInspiringLamePuns
cp -r config/parameter_configuration_mapping MixedInspiringLamePuns
cp -r config/data_utils MixedInspiringLamePuns
zip -r config/MixedInspiringLamePuns.zip MixedInspiringLamePuns
mv MixedInspiringLamePuns ../submissions