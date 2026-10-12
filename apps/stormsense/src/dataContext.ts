import { createContext, useContext } from "react";
import { LIVE_DATA } from "./data";

export const DataSourceContext = createContext(LIVE_DATA);
export const useDataSource = () => useContext(DataSourceContext);
