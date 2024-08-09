find . -name "MITIQ_ZNE_Linear" -exec sed -i 's/"ERROR_MITIGATION": null/"ERROR_MITIGATION":"MITIQ_Linear"/g' {}/IQP_Full-Pauli-CRX/IQP_Full-Pauli-CRX.json \;
find . -name "MITIQ_ZNE_Richardson" -exec sed -i 's/"ERROR_MITIGATION": null/"ERROR_MITIGATION":"MITIQ_Richardson"/g' {}/IQP_Full-Pauli-CRX/IQP_Full-Pauli-CRX.json \;
find . -name "M3" -exec sed -i 's/"ERROR_MITIGATION": null/"ERROR_MITIGATION":"M3"/g' {}/IQP_Full-Pauli-CRX/IQP_Full-Pauli-CRX.json \;
find . -name "TREX" -exec sed -i 's/"ERROR_MITIGATION": null/"ERROR_MITIGATION":"TREX"/g' {}/IQP_Full-Pauli-CRX/IQP_Full-Pauli-CRX.json \;
