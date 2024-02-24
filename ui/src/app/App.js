import './App.css';
import { ThemeContext, themes } from "../api/Theme";
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Welcome from "../components/pages/Welcome";
import Home from "../components/pages/Home";

function App() {
    return (
        <ThemeContext.Provider value={themes.light}>
            <>
                <Router>
                    <Routes>
                        <Route exact path="/" element={<Welcome />} />
                        <Route exact path="/home" element={<Home />} />

                    </Routes>
                </Router>
            </>
        </ThemeContext.Provider>
    );
}

export default App;
