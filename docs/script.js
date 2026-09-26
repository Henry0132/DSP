const copyButton = document.querySelector('#copy-citation');
if (navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    const status = document.querySelector('#copy-status');
    try {
      await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent.trim());
      status.textContent = 'BibTeX copied.';
    } catch {
      status.textContent = 'Copy unavailable. Please select and copy the citation above.';
    }
  });
}
