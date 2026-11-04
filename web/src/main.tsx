import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import '@fontsource/ibm-plex-mono/latin-600.css';
import { createRoot } from 'react-dom/client';

import { App } from './App.tsx';
import './styles.css';
import './screens.css';

const root = document.getElementById('root');
if (root) createRoot(root).render(<App />);
