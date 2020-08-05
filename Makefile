
#################################################################################
# COMMANDS                                                                      #
#################################################################################
sync_data_to_gcloud: model_location
	python3 -c 'from ml.filesutil import fake_upload; fake_upload()'

model_location:
	python3 -c 'from ml.filesutil import local_models_to_json; local_models_to_json()'