

verifyGeometry
verifyConnectivity

# Timing report
report_timing -max_paths 5 > ${design}.post_route.timing.rpt

# Power report
report_power -outfile ${design}.post_route.power.rpt

# Design report
summaryReport -nohtml -outfile ${design}.post_route.summary.rpt

