# Two-Echelon Multi-Depot Multi-Day Multi-Period Capacitated Vehicle Routing Problem with Simultaneous Pickup and Delivery (2E-MD-MD-MP-CVRP-SPD)

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

- 使用**中國沿岸地區真實顧客地理座標與需求資料**作為輸入，模擬多日動態配送環境。

## 📈 問題特性

- **二層次配送架構**：主倉庫 → 衛星站點 → 客戶
- **多倉庫、多日、多期間**設定
- **同步取貨與送貨**（Simultaneous Pickup and Delivery）
- **容量限制**與**配送時段考量**

## 🔍 結果與效益

- 成功降低配送總成本與車輛使用數量
- 相較基準演算法具更高解品質與穩定性
- 展現於大規模場景中的強健應用潛力
