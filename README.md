<img width="677" height="137" alt="圖片" src="https://github.com/user-attachments/assets/27a9ec42-3107-4084-8431-1c1df523974f" /># Two-Echelon Multi-Depot Multi-Day Multi-Period Capacitated Vehicle Routing Problem with Simultaneous Pickup and Delivery (2E-MD-MD-MP-CVRP-SPD)

本專案針對具**同時取送貨特性**的**二層次多倉庫多日多期間容量限制車輛路徑問題**（2E-MD-MD-MP-CVRP-SPD）提出一套優化演算法架構，能有效處理中國沿岸地區真實顧客資料，並於短時間內產出具品質的最適配送路徑。

## 🚀 專案目標

- 模擬實際物流配送情境，處理每日變動之顧客需求與位置。
- 運用演算法自動決定衛星站點位置、分群與配送路徑。
- 實作高效能演化式演算法，提升解的品質並降低總配送成本。

## 🧠 使用技術

- 語言：Python
- 核心演算法：
  1. **3D K-Means 顧客分群**
  2. **動態衛星定位策略**
  3. **Clarke & Wright 節省法則（CW Savings）**
  4. **NSGA-II 多目標演化演算法**

## 📂 資料來源

- 使用**中國沿岸地區真實顧客地理座標與需求資料**作為輸入，模擬多日動態配送環境。 ![中國沿岸顧客分布](images/china_customers.pdf)

## 📈 問題特性

- **二層次配送架構**：主倉庫 → 衛星站點 → 客戶
- **多倉庫、多日、多期間**設定
- **同步取貨與送貨**（Simultaneous Pickup and Delivery）
- **容量限制**與**配送時段考量**

## 🧬 基因演算法中的編碼與解碼策略

為有效應用基因演算法於 **二層次多倉庫多日多期間容量限制車輛路徑問題（2E-MD-MD-MP-CVRP-SPD）**，本研究設計了專屬的 **染色體編碼（Encoding）、解碼（Decoding）與本地搜尋(Local Search)** 機制，並結合交叉與突變策略以強化解空間探索。

---

### 📦 染色體編碼（Encoding）

- 每條染色體代表一組配送解。
- ![編碼](images/Encode.pdf)
- 基因片段對應顧客編號，並依據分群結果進行區隔：
- 編碼方式考量群組內顧客的路徑排序與分配關係，有助於後續以衛星據點為中心進行二層配送規劃

### 🔁 染色體解碼（Decoding）

- 將染色體轉換為具體配送路徑，範例如下：
- ![解碼](images/Decode.pdf)
- 每條路徑對應一部次級車輛由衛星據點出發，服務所屬群組顧客後返回據點。
- 解碼結果作為適應度函數評估的依據（例如路徑成本、車輛數量等）。

## 🔄 染色體交叉、突變設計與本地搜尋（Crossover & Mutation & Local Search）

為提升族群多樣性與演化品質，本研究於 NSGA-II 框架中導入下列遺傳操作機制：

### 🔁 有序交叉（Ordered Crossover, OX）
1. **隨機選取交叉區段**：自 `Parent1` 中擷取一段連續基因，例如 `5,6,7`。
2. **保留其餘基因順序**：從 `Parent2` 依序補上未出現在交叉段中的基因。
3. **生成子代**：產生無重複基因的合法染色體。 ![OX](imahes/OX.drawio.pdf)

### 🧬 隨機交換突變（Mutation）

- **操作方式**：隨機選擇染色體中的兩個基因並交換其位置。
- **目的**：避免早期收斂，增強搜尋多樣性。![Mutation](imahes/Mutation.drawio.pdf)

### 🔁 本地搜尋(Local Search)

- **Exchange 操作**：隨機選擇染色體中的兩個基因並交換其位置。![Exchange](imahes/Exchange.drawio.pdf)
- **2-opt 操作**：隨機選擇染色體中的一段基因序並顛倒其位置。 ![2-opt](imahes/2-opt.drawio.pdf)
- **Relocate 操作**：隨機選擇染色體中的一個基因並隨機插入其位置。![Relocate](imahes/Relocate.drawio.pdf)

## 🔍 結果與效益

- 成功降低配送總成本與車輛使用數量
- 相較基準演算法具更高解品質與穩定性
- 展現於大規模場景中的強健應用潛力
