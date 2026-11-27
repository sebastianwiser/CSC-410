# Earlier audit scope

The first executed serve audit reused the model loader's rank-completeness filter.
Its original numbers are retained here as historical evidence and logged as
`serve_data_audit` in MLflow. The final `results/serve_audit*` artifacts remove that
unnecessary filter and use every available valid serve-stat row. The final run is
logged as `serve_data_audit_all_available_stats`. Neither audit interprets the
observed ATP files as complete career records.
