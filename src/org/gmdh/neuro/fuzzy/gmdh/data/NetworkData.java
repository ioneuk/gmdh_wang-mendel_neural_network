package org.gmdh.neuro.fuzzy.gmdh.data;

import lombok.Data;
import lombok.RequiredArgsConstructor;

import java.util.List;

@RequiredArgsConstructor
@Data
public class NetworkData {

    List<DataEntry> dataEntries;

    public DataEntry getDataEntryWithNumber(int entryNumber) {
        return dataEntries.get(entryNumber);
    }
}
