package org.gmdh.neuro.fuzzy.gmdh.data;

import lombok.Data;
import lombok.RequiredArgsConstructor;

import java.util.List;

@RequiredArgsConstructor
@Data
public class NetworkData {

    private List<DataEntry> dataEntries;

    public NetworkData(List<DataEntry> dataEntries) {
        this.dataEntries = dataEntries;
    }

    public DataEntry getDataEntryWithNumber(int entryNumber) {
        return dataEntries.get(entryNumber);
    }

    public int getRegressorsCount() {
        return dataEntries.get(0).getRegressors().length;
    }

    public int getDataEntriesCount() {
        return dataEntries.size();
    }

    public double getMaxOutput() {
        return dataEntries.stream().map(DataEntry::getResult).mapToDouble(Double::valueOf).max().getAsDouble();
    }
}
